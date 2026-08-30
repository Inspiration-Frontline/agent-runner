import time
import urllib.error
import urllib.request
import webbrowser


def open_docs_when_ready(server_port: int, timeout_seconds: float = 30.0) -> None:
    """Open the local API documentation after the development server accepts connections.

    Args:
        server_port: Local HTTP port on which the documentation endpoint is served.
        timeout_seconds: Maximum duration in seconds to wait for local startup checks.
    """
    health_url: str = f"http://127.0.0.1:{server_port}/health"
    docs_url: str = f"http://127.0.0.1:{server_port}/docs"
    deadline: float = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=0.5) as response:
                if response.status == 200:
                    webbrowser.open(docs_url)
                    return
        except (OSError, urllib.error.URLError):
            time.sleep(0.2)
