from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Tuple, Any
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from envfile import GROQ_API_KEY
from fetch_source_code import fetch_page_source_with_url
import json
from pydantic import BaseModel, Field
from urllib.parse import urljoin, urlparse, urldefrag
from collections import deque
from typing import Set
import concurrent.futures
import re
import os


class SourceCode(TypedDict):
    base_url: str
    page_source: str
    endpoints: List[str]


class SourceAnalysis(BaseModel):
    endpoints: List[str] = Field(
        default_factory=list,
        description="List of endpoints with relative URLs and parameters"
    )


def fetch_source_node(state: SourceCode) -> SourceCode:
    # Use the provided base_url from the state. Do not use a hardcoded default here.
    url = state.get("base_url")
    html_source, content_type = fetch_page_source_with_url(url)
    
    print(f"🔍 Fetching source from: {url} (Content-Type: {content_type})")
    print(f"✅ Source code fetched ({len(html_source)} characters)")
    
    return {
        "base_url": url,
        "page_source": html_source,
        "endpoints": []
    }


def analyze_and_print_source_code(state: SourceCode) -> SourceCode:
    """
    Analyze source code and print endpoints with full URLs
    """
    page_source = state["page_source"]
    base_url = state["base_url"]
    
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0
    )
    structured_llm = llm.with_structured_output(SourceAnalysis)

    # Performance/configuration flags
    FAST_MODE = os.getenv("RECON_FAST_MODE", "1") == "1"  # when True, use regex-first (no LLM) for quick extraction
    CONCURRENCY = int(os.getenv("RECON_CONCURRENCY", "8"))

    # Feature flags
    FOLLOW_ALL = os.getenv("RECON_FOLLOW_ALL", "0") == "1"

    system_prompt = """
    You are a web application security analyst. Analyze HTML source code and extract endpoints (pages/forms/APIs) with ALL parameters (e.g., /search.php?q=test&category=books).

    Return ONLY relative paths. For endpoints, include ALL query parameters.
    """

    user_prompt = f"""
    BASE URL: {base_url}
    
    HTML SOURCE (truncated):
    {page_source[:10000]}
    
    Extract endpoints with parameters:
    """
    
    
    # Discover endpoints reachable from the found endpoints (BFS, limited)
    def to_full_url(endpoint: str, base: str) -> str:
        if endpoint.startswith(('http://', 'https://')):
            return endpoint
        if endpoint.startswith('/'):
            return urljoin(base.rsplit('/', 1)[0] if '/' in base else base, endpoint)
        return urljoin(base, endpoint)


    def normalize_url(u: str) -> str:
        # remove fragment and normalize
        try:
            nofrag, _ = urldefrag(u)
            return nofrag
        except Exception:
            return u


    def same_domain(base: str, u: str) -> bool:
        try:
            return urlparse(base).netloc.lower() == urlparse(u).netloc.lower()
        except Exception:
            return False


    STATIC_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.css', '.ico', '.woff', '.woff2', '.ttf', '.map', '.eot'}


    def is_static_asset(u: str) -> bool:
        p = urlparse(u).path.lower()
        for ext in STATIC_EXTENSIONS:
            if p.endswith(ext):
                return True
        return False


    def quick_html_extract_endpoints(html: str, llm_structured=None) -> List[str]:
        """Lightweight HTML scanner to extract href/src/action-like endpoints quickly without LLM.

        Returns a list of candidate relative or absolute endpoint strings.
        """
        candidates: Set[str] = set()
        try:
            # href/src
            for m in re.findall(r"(?:href|src)=['\"]([^'\"]+)['\"]", html, flags=re.IGNORECASE):
                candidates.add(m)
            # form actions
            for m in re.findall(r"<form[^>]+action=['\"]([^'\"]+)['\"]", html, flags=re.IGNORECASE):
                candidates.add(m)
            # JS-like strings inside HTML
            for m in re.findall(r"['\"]((?:/|https?:)//?[^'\"]+\\.php[^'\"]*)['\"]", html, flags=re.IGNORECASE):
                candidates.add(m)

            # Extract inline <script> blocks and use AI for JS extraction when available
            for script in re.findall(r"<script[^>]*>(.*?)</script>", html, flags=re.IGNORECASE | re.DOTALL):
                if llm_structured:
                    try:
                        for s_ep in ai_extract_endpoints_from_js(script, llm_structured):
                            candidates.add(s_ep)
                    except Exception:
                        for m in re.findall(r"['\"]((?:https?:\\/\\/|\\/)[^'\"\s<>]+)['\"]", script):
                            candidates.add(m)
                else:
                    for m in re.findall(r"['\"]((?:https?:\\/\\/|\\/)[^'\"\s<>]+)['\"]", script):
                        candidates.add(m)
        except Exception:
            return []

        # If FOLLOW_ALL is enabled, accept most non-trivial links (skip mailto:, javascript:, fragments)
        if FOLLOW_ALL:
            filtered: List[str] = []
            for c in candidates:
                if not c:
                    continue
                lc = c.lower()
                if lc.startswith('#'):
                    continue
                if lc.startswith('javascript:') or lc.startswith('mailto:'):
                    continue
                filtered.append(c)
            return filtered

        # Heuristic filter: prefer likely endpoints and skip static assets
        filtered: List[str] = []
        for c in candidates:
            if not c:
                continue
            lc = c.lower()
            if c.startswith('#'):
                continue
            if any(x in lc for x in ['.php', '.asp', '.aspx', '/api/', '.json', '?']) or c.startswith('/'):
                filtered.append(c)
        return filtered


    def regex_extract_endpoints_from_js(js_content: str) -> List[str]:
        """Extract URL-like strings from JS source using regex heuristics."""
        candidates = re.findall(r"['\"]((?:https?:\\/\\/|\\/)[^'\"\s<>]+)['\"]", js_content)
        filtered: List[str] = []
        for c in set(candidates):
            lc = c.lower()
            if any(x in lc for x in ['.php', '.asp', '.aspx', '.json', '/api/', '?']) or c.startswith('/'):
                filtered.append(c)
        return filtered


    def ai_extract_endpoints_from_js(js_content: str, llm_structured) -> List[str]:
        """Use the structured LLM to extract relative endpoints from JS content only.

        Returns a list of relative paths (starting with "/" or path segments) only.
        The model must return structured output matching SourceAnalysis (endpoints list) and no extra text.
        """
        if not llm_structured:
            return regex_extract_endpoints_from_js(js_content)

        system_prompt_js = """
You are a web application analyst. From the provided JavaScript code, extract ALL endpoint paths used by network calls (fetch, axios, XMLHttpRequest.open, new WebSocket, or other direct URL usages).

Rules:
- Return ONLY a JSON object matching the schema: {"endpoints": ["/path1", "/path2?x=y"]}.
- Return relative paths only (starting with '/' or no scheme/domain). Do NOT return full URLs with a different domain.
- Handle string concatenation and template literals (e.g., `/api/` + id or `/api/${id}`) by returning the path template if possible (keep placeholders as-is).
- Do NOT include explanations, commentary, or any extra keys.
"""

        try:
            res: SourceAnalysis = llm_structured.invoke([
                SystemMessage(content=system_prompt_js),
                HumanMessage(content=f"JS SOURCE (truncated):\n{js_content[:15000]}")
            ])
            # Ensure only relative paths are returned
            out: List[str] = []
            for e in (res.endpoints or []):
                if not e:
                    continue
                e = e.strip()
                # Only keep relative paths (no scheme)
                if e.startswith('http://') or e.startswith('https://'):
                    # Convert to relative if same domain? We drop absolute to obey constraint: extract relative only
                    continue
                out.append(e)
            return out
        except Exception:
            return regex_extract_endpoints_from_js(js_content)


    def extract_from_json_text(text: str) -> List[str]:
        """Parse JSON and recursively collect URL-like strings."""
        found: Set[str] = set()
        try:
            obj = json.loads(text)
        except Exception:
            # fallback to regex scan
            return regex_extract_endpoints_from_js(text)

        def walk(o: Any):
            if isinstance(o, str):
                for u in re.findall(r"(https?:\\/\\/[^\s'\"<>]+|\\/[^\s'\"<>]+\.[a-z]{2,}[^\s'\"<>]*)", o):
                    # heuristic filter
                    if any(x in u.lower() for x in ['.php', '/api/', '.json', '?'] ) or u.startswith('/'):
                        found.add(u)
            elif isinstance(o, list):
                for i in o:
                    walk(i)
            elif isinstance(o, dict):
                for k, v in o.items():
                    walk(k)
                    walk(v)

        walk(obj)
        return list(found)


    def extract_endpoints_from_content(content: str, content_type: str, base: str) -> List[str]:
        """Dispatch extraction based on Content-Type header or heuristics."""
        if not content_type:
            # guess from content
            ct = content.strip()[:20].lower()
            if ct.startswith('{') or ct.startswith('['):
                content_type = 'application/json'
            elif '<script' in content.lower() or '<html' in content.lower():
                content_type = 'text/html'
            else:
                content_type = 'text/plain'

        if 'json' in content_type:
            return extract_from_json_text(content)
        if 'javascript' in content_type:
            # Prefer AI-assisted JS extraction for better accuracy, fall back to regex
            try:
                return ai_extract_endpoints_from_js(content, structured_llm)
            except Exception:
                return regex_extract_endpoints_from_js(content)
        # default: treat as HTML
        return quick_html_extract_endpoints(content, structured_llm)

    def discover_and_print_recursive(start_base: str, initial_endpoints: List[str], llm_structured) -> Set[str]:
        """Fetch each discovered endpoint, run the same LLM analysis on it, and print any new endpoints found.

        This does a breadth-first traversal over discovered endpoints up to max_visits total pages to avoid infinite loops.
        Returns the set of all discovered full endpoint URLs (excluding the initial ones).
        """
        # Use frontier-based batch processing with a ThreadPoolExecutor to fetch pages concurrently
        visited: Set[str] = set()
        discovered_new: Set[str] = set()

        # Normalize initial endpoints into a frontier list (filter static assets and out-of-domain)
        frontier: List[str] = []
        for ep in initial_endpoints:
            try:
                full = normalize_url(to_full_url(ep, start_base))
            except Exception:
                continue
            if not same_domain(start_base, full):
                continue
            if is_static_asset(full):
                continue
            if full not in visited:
                frontier.append(full)
                visited.add(full)

        visits = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            # Continue while there are items in the frontier. No enforced upper bound here.
            while frontier:
                # take up to CONCURRENCY items from frontier
                batch = frontier[:CONCURRENCY]
                frontier = frontier[CONCURRENCY:]

                # Submit fetches concurrently
                future_to_url = {executor.submit(fetch_page_source_with_url, url): url for url in batch}
                for fut in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[fut]
                    visits += 1
                    print(f"\n➡️ Exploring endpoint ({visits}): {url}")
                    try:
                        src, content_type = fut.result()
                    except Exception as e:
                        print(f"  ⚠️ Failed to fetch {url}: {e}")
                        continue
                    # Extract endpoints based on content-type (fast heuristics)
                    eps = extract_endpoints_from_content(src, content_type, url)
                    new_eps = []
                    for ep in eps:
                        try:
                            full_ep = normalize_url(to_full_url(ep, url))
                        except Exception:
                            continue
                        # enforce same-domain and skip static assets
                        if not same_domain(start_base, full_ep):
                            continue
                        if is_static_asset(full_ep):
                            continue
                        if full_ep not in visited:
                            visited.add(full_ep)
                            frontier.append(full_ep)
                            discovered_new.add(full_ep)
                            new_eps.append(full_ep)
                            print(f"  ⚡ Fast-found endpoint: {full_ep}")

                    # If FAST_MODE is False and no endpoints were found quickly, fall back to LLM for deeper analysis
                    if (not FAST_MODE) and (not new_eps):
                        try:
                            prompt = f"BASE URL: {url}\n\nHTML SOURCE (truncated):\n{src[:10000]}\n\nExtract endpoints with parameters:"
                            try:
                                res: SourceAnalysis = llm_structured.invoke([
                                    SystemMessage(content=system_prompt),
                                    HumanMessage(content=prompt)
                                ])
                            except Exception as e:
                                print(f"  ⚠️ LLM analysis failed for {url} (fallback to fast extractor): {e}")
                                res = SourceAnalysis(endpoints=quick_html_extract_endpoints(src, llm_structured))
                        except Exception as e:
                            print(f"  ⚠️ LLM analysis failed for {url}: {e}")
                            continue

                        for ep in set(res.endpoints or []):
                            if not ep:
                                continue
                            try:
                                full_ep = normalize_url(to_full_url(ep, url))
                            except Exception:
                                continue
                            if not same_domain(start_base, full_ep):
                                continue
                            if is_static_asset(full_ep):
                                continue
                            if full_ep not in visited:
                                visited.add(full_ep)
                                frontier.append(full_ep)
                                discovered_new.add(full_ep)
                                print(f"  ✅ New endpoint discovered: {full_ep}")

        if not discovered_new:
            print("\nNo additional endpoints discovered from fetched endpoints.")

        # Return the full set of visited endpoints so the caller can report counts
        return visited

    # Initial analysis: use LLM unless FAST_MODE is enabled; fall back to fast extractor on error
    # If FAST_MODE is enabled, prefer the lightweight regex extractor and avoid LLM usage.
    if FAST_MODE:
        print("⚡ FAST_MODE enabled — using lightweight extractor instead of the LLM for initial analysis")
        extracted = quick_html_extract_endpoints(page_source, structured_llm)
        result = SourceAnalysis(endpoints=extracted)
    else:
        try:
            result: SourceAnalysis = structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
        except Exception as e:
            print(f"⚠️ LLM analysis failed (falling back to fast extractor): {e}")
            extracted = quick_html_extract_endpoints(page_source, structured_llm)
            result = SourceAnalysis(endpoints=extracted)

    # Convert to full URLs and print
    print("\n" + "="*80)
    print("📊 WEB APPLICATION ANALYSIS RESULTS")
    print("="*80)
    
    # Print endpoints with full URLs including ALL parameters
    if result.endpoints:
        print("\n🔗 ENDPOINTS (Full URLs with Parameters):")
        print("-"*40)
        for endpoint in set(result.endpoints):
            if endpoint:
                # Convert to full URL
                if endpoint.startswith(('http://', 'https://')):
                    full_url = endpoint
                elif endpoint.startswith('/'):
                    full_url = urljoin(base_url.rsplit('/', 1)[0] if '/' in base_url else base_url, endpoint)
                else:
                    full_url = urljoin(base_url, endpoint)
                
                # Print with parameters in the URL
                print(f"• {full_url}")
    else:
        print("\n❌ No endpoints found")
    
    print("="*80)

    # Run recursive discovery using the structured LLM
    try:
        discovered = discover_and_print_recursive(base_url, result.endpoints or [], structured_llm)
    except Exception as e:
        print(f"Error during recursive discovery: {e}")
        discovered = set()

    # Print a concise summary of what we found
    try:
        total_found = len(discovered)
    except Exception:
        total_found = 0

    print("\n" + "="*40)
    print(f"Summary: total unique endpoints discovered (including initial): {total_found}")
    print("="*40)

    return {
        "base_url": base_url,
        "page_source": page_source,
        "endpoints": result.endpoints
    }


def build_graph():
    graph = StateGraph(SourceCode)

    graph.add_node("fetch_source", fetch_source_node)
    graph.add_node("analyze_and_print", analyze_and_print_source_code)

    graph.add_edge(START, "fetch_source")
    graph.add_edge("fetch_source", "analyze_and_print")
    graph.add_edge("analyze_and_print", END)

    return graph.compile()


if __name__ == "__main__":
    # Allow the user to pass a URL via CLI or be prompted interactively.
    import argparse

    parser = argparse.ArgumentParser(description="Analyze a target web page for endpoints.")
    parser.add_argument("--url", "-u", help="Target base URL to analyze (e.g. https://example.com/index.php)")
    parser.add_argument("--follow-all", action="store_true", help="Follow all non-trivial links (overrides endpoint heuristics). May discover many URLs.")
    args = parser.parse_args()

    target_url = args.url
    if not target_url:
        try:
            target_url = input("Enter target URL: ").strip()
        except (EOFError, KeyboardInterrupt):
            target_url = None

    if not target_url:
        print("No target URL provided. Exiting.")
        raise SystemExit(1)

    # Note: max visits should be controlled via the RECON_MAX_VISITS environment variable.
    if args.follow_all:
        os.environ["RECON_FOLLOW_ALL"] = "1"

    app = build_graph()

    initial_state: SourceCode = {
        "base_url": target_url,
        "page_source": "",
        "endpoints": []
    }

    final_state = app.invoke(initial_state)
