import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, urljoin, urlparse

import httpx
from agents import function_tool

from agent_runner.config import get_settings


@dataclass(frozen=True)
class _FetchedPage:
    """Safe page body and final URL returned by the fetch boundary."""

    body: str
    final_url: str


@dataclass
class _SearchResult:
    """One normalized DuckDuckGo result before fetching its destination."""

    url: str
    title: str = ""
    snippet: str = ""


class _WebSearchClient:
    async def search(self, query: str) -> dict[str, object]:
        """Search DuckDuckGo and fetch readable content from the bounded result set."""
        normalized = query.strip()
        if not normalized:
            raise ValueError("Search query cannot be blank.")
        settings = get_settings()
        async with httpx.AsyncClient(
            timeout=settings.tool_http_timeout_seconds,
            headers={"User-Agent": "AgentBreaker/0.0.1 (+local development)"},
            follow_redirects=False,
        ) as client:
            search_response = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": normalized},
            )
            search_response.raise_for_status()
            parser = _DuckDuckGoResultParser(settings.web_search_max_results)
            parser.feed(search_response.text)
            parser.close()
            raw_results = parser.results
            if not raw_results:
                raise ValueError(f"DuckDuckGo returned no results for: {normalized}")
            pages = await asyncio.gather(
                *(self._fetch_result(client, result) for result in raw_results),
            )

        usable_pages = [page for page in pages if page.get("content")]
        if not usable_pages:
            raise ValueError("Search succeeded, but no result page produced readable content.")
        return {"query": normalized, "results": pages}

    async def _fetch_result(
        self,
        client: httpx.AsyncClient,
        result: _SearchResult,
    ) -> dict[str, object]:
        """Fetch and sanitize one result page while preserving per-page errors."""
        item: dict[str, object] = {
            "title": result.title,
            "url": result.url,
            "snippet": result.snippet,
            "content": "",
            "error": "",
        }
        try:
            fetched_page = await self._safe_get(client, result.url)
            parser = _VisibleTextParser()
            parser.feed(fetched_page.body)
            parser.close()
            content = parser.get_text()
            item["url"] = fetched_page.final_url
            item["content"] = content or ""
            if not item["content"]:
                item["error"] = "No readable page content was extracted."
        except Exception as error:
            item["error"] = str(error)
        return item

    async def _safe_get(self, client: httpx.AsyncClient, url: str) -> _FetchedPage:
        """Follow safe public redirects and enforce content, byte, and redirect limits."""
        settings = get_settings()
        current_url = url
        for _ in range(settings.web_fetch_max_redirects + 1):
            await self._require_public_http_url(current_url)
            async with client.stream("GET", current_url) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("Page redirect did not include a location.")
                    current_url = urljoin(current_url, location)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "text/html" not in content_type and "text/plain" not in content_type:
                    raise ValueError(f"Unsupported page content type: {content_type or 'unknown'}")
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    body.extend(chunk)
                    if len(body) > settings.web_fetch_max_bytes:
                        raise ValueError("Page exceeded the network safety byte limit.")
                return _FetchedPage(
                    body=body.decode(response.encoding or "utf-8", errors="replace"),
                    final_url=str(response.url),
                )
        raise ValueError("Page exceeded the redirect limit.")

    @staticmethod
    async def _require_public_http_url(url: str) -> None:
        """Reject non-HTTP, private, loopback, or otherwise non-public result destinations."""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Search result URL must use public HTTP or HTTPS.")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = await asyncio.get_running_loop().getaddrinfo(
            parsed.hostname,
            port,
            type=socket.SOCK_STREAM,
        )
        if not addresses:
            raise ValueError("Search result hostname did not resolve.")
        for address in addresses:
            ip = ipaddress.ip_address(address[4][0])
            if not ip.is_global:
                raise ValueError("Search result resolved to a non-public address.")


@function_tool(failure_error_function=None)
async def search_web(query: str) -> dict[str, object]:
    """Search the public web and return fetched page content with source URLs.

    Args:
        query: Search query to send to DuckDuckGo.
    """
    return await _WebSearchClient().search(query)


class _DuckDuckGoResultParser(HTMLParser):
    def __init__(self, limit: int) -> None:
        """Initialize an HTML parser that captures at most the configured result count."""
        super().__init__(convert_charrefs=True)
        self.limit = limit
        self.results: list[_SearchResult] = []
        self._current: _SearchResult | None = None
        self._capture: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Start collecting a result title or snippet when the expected class is encountered."""
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "a" and "result__a" in classes and len(self.results) < self.limit:
            href = attributes.get("href") or ""
            parsed = urlparse(href)
            redirected = parse_qs(parsed.query).get("uddg")
            self._current = _SearchResult(url=redirected[0] if redirected else href)
            self._capture = "title"
        elif self._current is not None and "result__snippet" in classes:
            self._capture = "snippet"

    def handle_endtag(self, tag: str) -> None:
        """Finalize a result when its title or snippet container closes."""
        if tag == "a" and self._current is not None and self._capture == "title":
            self._capture = None
        elif tag in {"a", "div"} and self._current is not None and self._capture == "snippet":
            self.results.append(self._current)
            self._current = None
            self._capture = None

    def handle_data(self, data: str) -> None:
        """Append visible text to the currently captured result field."""
        if self._current is not None and self._capture == "title":
            self._current.title += data
        elif self._current is not None and self._capture == "snippet":
            self._current.snippet += data

    def close(self) -> None:
        """Flush a partially closed result before releasing parser state."""
        super().close()
        if self._current is not None and len(self.results) < self.limit:
            self.results.append(self._current)
            self._current = None


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        """Initialize an HTML parser that ignores executable and non-visible elements."""
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Enter an ignored region for scripts, styles, SVG, and noscript content."""
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        """Leave an ignored region when its closing tag is observed."""
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        """Normalize and retain visible text fragments from the current HTML document."""
        if not self._ignored_depth:
            normalized = " ".join(data.split())
            if normalized:
                self._parts.append(normalized)

    def get_text(self) -> str:
        """Return normalized visible text joined into model-readable lines."""
        # TODO: Apply a shared semantic context trimmer after AgentBreaker defines a unified
        # replay/tool/RAG context policy. Extracted text is retained within the network cap.
        return "\n".join(self._parts)
