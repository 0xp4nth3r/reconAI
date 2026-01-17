from urllib import request, error
import threading


def fetch_page_source_with_url(url: str, timeout: int = 10):
    """Fetch a URL and return a tuple (decoded_text, content_type_header).

    Uses a simple User-Agent and returns decoded text (charset-aware when possible).
    Raises exceptions on network errors.
    """
    req = request.Request(url, headers={
        'User-Agent': 'recon-phase1/1.0',
        'Accept': '*/*'
    })
    resp = request.urlopen(req, timeout=timeout)
    content_type = resp.getheader('Content-Type') or ''
    raw = resp.read()

    # Try to detect charset in Content-Type
    charset = 'utf-8'
    if 'charset=' in content_type:
        try:
            charset = content_type.split('charset=')[-1].split(';')[0].strip()
        except Exception:
            charset = 'utf-8'

    try:
        text = raw.decode(charset, errors='replace')
    except Exception:
        try:
            text = raw.decode('utf-8', errors='replace')
        except Exception:
            text = raw.decode('latin-1', errors='replace')

    return text, content_type


# Simple in-memory thread-safe cache to avoid re-fetching the same URL multiple
# times during a single run. This helps when discovery revisits pages or when
# multiple threads request the same resource.
_FETCH_CACHE = {}
_FETCH_CACHE_LOCK = threading.Lock()


def fetch_page_source_with_url_cached(url: str, timeout: int = 10, use_cache: bool = True):
    """Wrapper around fetch_page_source_with_url that caches results in-memory.

    Returns the same (text, content_type) tuple. Set use_cache=False to force
    a fresh network request.
    """
    if use_cache:
        with _FETCH_CACHE_LOCK:
            if url in _FETCH_CACHE:
                return _FETCH_CACHE[url]

    result = fetch_page_source_with_url(url, timeout=timeout)

    if use_cache:
        with _FETCH_CACHE_LOCK:
            _FETCH_CACHE[url] = result

    return result


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        src, ct = fetch_page_source_with_url(sys.argv[1])
        print(f"Content-Type: {ct}\n\n{src[:2000]}")
    else:
        print("Usage: fetch_source_code.py <url>")

