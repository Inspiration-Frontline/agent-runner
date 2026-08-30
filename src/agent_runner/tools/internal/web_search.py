import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import ParseResult, parse_qs, urljoin, urlparse

import httpx
from agents import function_tool

from agent_runner.config import Settings, get_settings


@dataclass(frozen=True)
class _FetchedPage:
    """Safe page body and final URL returned by the fetch boundary."""

    body: str
    """Decoded page body bounded by the configured byte limit."""
    final_url: str
    """Final URL after any allowed redirects."""


@dataclass
class _SearchResult:
    """One normalized DuckDuckGo result before fetching its destination."""

    url: str
    """Result URL returned by the search provider."""
    title: str = ""
    """Result title shown to the Agent."""
    snippet: str = ""
    """Search-provider summary text for the result."""


class _WebSearchClient:
    async def search(self, query: str) -> dict[str, object]:
        """Search DuckDuckGo and fetch readable content from the bounded result set.

        Args:
            query: Natural-language query sent to the retrieval boundary.

        Returns:
            Bounded search results with readable page excerpts and source URLs.
        """
        normalized: str = query.strip()
        if not normalized:
            raise ValueError("Search query cannot be blank.")
        settings: Settings = get_settings()
        async with httpx.AsyncClient(
            timeout=settings.tool_http_timeout_seconds,
            headers={"User-Agent": "AgentBreaker/0.0.1 (+local development)"},
            follow_redirects=False,
        ) as client:
            search_response: httpx.Response = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": normalized},
            )
            search_response.raise_for_status()
            parser: _DuckDuckGoResultParser = _DuckDuckGoResultParser(settings.web_search_max_results)
            parser.feed(search_response.text)
            parser.close()
            raw_results: list[_SearchResult] = parser.results
            if not raw_results:
                raise ValueError(f"DuckDuckGo returned no results for: {normalized}")
            pages: list[dict[str, object]] = await asyncio.gather(
                *(self._fetch_result(client, result) for result in raw_results),
            )

        usable_pages: list[dict[str, object]] = [page for page in pages if page.get("content")]
        if not usable_pages:
            raise ValueError("Search succeeded, but no result page produced readable content.")
        return {"query": normalized, "results": pages}

    async def _fetch_result(
        self,
        client: httpx.AsyncClient,
        result: _SearchResult,
    ) -> dict[str, object]:
        """Fetch and sanitize one result page while preserving per-page errors.

        Args:
            client: Asynchronous client used for the external boundary call.
            result: Operation result to normalize, trace, or persist.

        Returns:
            Fetched and sanitize one result page while preserving per-page errors.
        """
        item: dict[str, object] = {
            "title": result.title,
            "url": result.url,
            "snippet": result.snippet,
            "content": "",
            "error": "",
        }
        try:
            fetched_page: _FetchedPage = await self._safe_get(client, result.url)
            parser: _VisibleTextParser = _VisibleTextParser()
            parser.feed(fetched_page.body)
            parser.close()
            content: str = parser.get_text()
            item["url"] = fetched_page.final_url
            item["content"] = content or ""
            if not item["content"]:
                item["error"] = "No readable page content was extracted."
        except Exception as error:
            item["error"] = str(error)
        return item

    async def _safe_get(self, client: httpx.AsyncClient, url: str) -> _FetchedPage:
        """Follow safe public redirects and enforce content, byte, and redirect limits.

        Args:
            client: Asynchronous client used for the external boundary call.
            url: Absolute HTTP or HTTPS URL to validate or request.

        Returns:
            A bounded fetched page, or a typed fetch failure when policy rejects the URL.
        """
        settings: Settings = get_settings()
        current_url: str = url
        for _ in range(settings.web_fetch_max_redirects + 1):
            await self._require_public_http_url(current_url)
            async with client.stream("GET", current_url) as response:
                if response.is_redirect:
                    location: str | None = response.headers.get("location")
                    if not location:
                        raise ValueError("Page redirect did not include a location.")
                    current_url = urljoin(current_url, location)
                    continue
                response.raise_for_status()
                content_type: str = response.headers.get("content-type", "").lower()
                if "text/html" not in content_type and "text/plain" not in content_type:
                    raise ValueError(f"Unsupported page content type: {content_type or 'unknown'}")
                body: bytearray = bytearray()
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
        """Reject non-HTTP, private, loopback, or otherwise non-public result destinations.

        Args:
            url: Absolute HTTP or HTTPS URL to validate or request.
        """
        parsed: ParseResult = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Search result URL must use public HTTP or HTTPS.")
        port: int = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses: list[
            tuple[socket.AddressFamily, socket.SocketKind, int, str, tuple[str, int] | tuple[str, int, int, int]]
        ] = await asyncio.get_running_loop().getaddrinfo(
            parsed.hostname,
            port,
            type=socket.SOCK_STREAM,
        )
        if not addresses:
            raise ValueError("Search result hostname did not resolve.")
        for address in addresses:
            ip: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.ip_address(address[4][0])
            if not ip.is_global:
                raise ValueError("Search result resolved to a non-public address.")


@function_tool(failure_error_function=None)
async def search_web(query: str) -> dict[str, object]:
    """Search the public web and return fetched page content with source URLs.

    Args:
        query: Search query to send to DuckDuckGo.

    Returns:
        Bounded public-web results with fetched page content and source URLs.
    """
    return await _WebSearchClient().search(query)


class _DuckDuckGoResultParser(HTMLParser):
    """Extract a bounded ordered result set from DuckDuckGo HTML.

    Attributes:
        limit: Maximum number of search results retained from one response.
        results: Completed normalized results in document order.
        _current: Result currently receiving title/snippet text, if any.
        _capture: Name of the current result field receiving parser text.
    """

    def __init__(self, limit: int) -> None:
        """Initialize an HTML parser that captures at most the configured result count.

        Args:
            limit: Maximum number of parser results retained.
        """
        super().__init__(convert_charrefs=True)
        self.limit = limit
        self.results: list[_SearchResult] = []
        self._current: _SearchResult | None = None
        self._capture: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Start collecting a result title or snippet when the expected class is encountered.

        Args:
            tag: HTML element name supplied by the parser callback.
            attrs: HTML attribute names and values supplied by the parser callback.
        """
        attributes: dict[str, str | None] = dict(attrs)
        classes: set[str] = set((attributes.get("class") or "").split())
        if tag == "a" and "result__a" in classes and len(self.results) < self.limit:
            href: str = attributes.get("href") or ""
            parsed: ParseResult = urlparse(href)
            redirected: list[str] | None = parse_qs(parsed.query).get("uddg")
            self._current = _SearchResult(url=redirected[0] if redirected else href)
            self._capture = "title"
        elif self._current is not None and "result__snippet" in classes:
            self._capture = "snippet"

    def handle_endtag(self, tag: str) -> None:
        """Finalize a result when its title or snippet container closes.

        Args:
            tag: HTML element name supplied by the parser callback.
        """
        if tag == "a" and self._current is not None and self._capture == "title":
            self._capture = None
        elif tag in {"a", "div"} and self._current is not None and self._capture == "snippet":
            self.results.append(self._current)
            self._current = None
            self._capture = None

    def handle_data(self, data: str) -> None:
        """Append visible text to the currently captured result field.

        Args:
            data: External configuration payload to validate and normalize.
        """
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
    """Collect normalized page text while excluding executable and non-visible elements.

    Attributes:
        _ignored_depth: Nesting depth inside ignored HTML elements.
        _parts: Visible normalized text fragments retained in document order.
    """

    def __init__(self) -> None:
        """Initialize an HTML parser that ignores executable and non-visible elements."""
        super().__init__(convert_charrefs=True)
        self._ignored_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Enter an ignored region for scripts, styles, SVG, and noscript content.

        Args:
            tag: HTML element name supplied by the parser callback.
            attrs: HTML attribute names and values supplied by the parser callback.
        """
        if tag in {"script", "style", "noscript", "svg"}:
            self._ignored_depth += 1

    def handle_endtag(self, tag: str) -> None:
        """Leave an ignored region when its closing tag is observed.

        Args:
            tag: HTML element name supplied by the parser callback.
        """
        if tag in {"script", "style", "noscript", "svg"} and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        """Normalize and retain visible text fragments from the current HTML document.

        Args:
            data: External configuration payload to validate and normalize.
        """
        if not self._ignored_depth:
            normalized: str = " ".join(data.split())
            if normalized:
                self._parts.append(normalized)

    def get_text(self) -> str:
        """Return normalized visible text joined into model-readable lines.

        Returns:
            return normalized visible text joined into model-readable lines.
        """
        # TODO: Apply a shared semantic context trimmer after AgentBreaker defines a unified
        # replay/tool/RAG context policy. Extracted text is retained within the network cap.
        return "\n".join(self._parts)
