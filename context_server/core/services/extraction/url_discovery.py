"""
URL Discovery Service for intelligent web crawling.

Provides sitemap parsing, robots.txt compliance, and URL prioritization
capabilities for enhanced crawling performance.
"""

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp

from context_server.core.services.extraction.utils import URLUtils

logger = logging.getLogger(__name__)


@dataclass
class URLInfo:
    """Information about a discovered URL."""

    url: str
    priority: float = 1.0
    last_modified: Optional[datetime] = None
    change_frequency: Optional[str] = None
    source: str = "unknown"  # "sitemap", "robots", "crawl"
    relevance_score: float = 1.0


@dataclass
class SitemapInfo:
    """Information about a discovered sitemap."""

    url: str
    last_modified: Optional[datetime] = None
    urls_count: int = 0


class RobotsTxtParser:
    """Parser for robots.txt files to ensure crawling compliance."""

    def __init__(self, robots_content: str, user_agent: str = "*"):
        self.robots_content = robots_content
        self.user_agent = user_agent
        self.disallowed_paths: Set[str] = set()
        self.crawl_delay: Optional[float] = None
        self.sitemaps: List[str] = []
        self._parse()

    def _parse(self):
        """Parse robots.txt content."""
        current_user_agent = None
        applies_to_us = False

        for line in self.robots_content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.lower().startswith("user-agent:"):
                current_user_agent = line.split(":", 1)[1].strip()
                applies_to_us = (
                    current_user_agent == "*"
                    or current_user_agent.lower() == self.user_agent.lower()
                )

            elif applies_to_us:
                if line.lower().startswith("disallow:"):
                    path = line.split(":", 1)[1].strip()
                    if path:
                        self.disallowed_paths.add(path)

                elif line.lower().startswith("crawl-delay:"):
                    try:
                        self.crawl_delay = float(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass

            # Sitemaps apply to all user agents
            if line.lower().startswith("sitemap:"):
                sitemap_url = line.split(":", 1)[1].strip()
                self.sitemaps.append(sitemap_url)

    def is_allowed(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        parsed = urlparse(url)
        path = parsed.path

        for disallowed in self.disallowed_paths:
            if disallowed == "/":
                return False
            elif path.startswith(disallowed):
                return False

        return True


class SitemapParser:
    """Parser for XML sitemaps to discover URLs efficiently."""

    async def parse_sitemap(
        self, sitemap_url: str
    ) -> Tuple[List[URLInfo], List[SitemapInfo]]:
        """Parse a sitemap and return URLs and nested sitemaps."""
        urls = []
        nested_sitemaps = []

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(sitemap_url, timeout=30) as response:
                    if response.status != 200:
                        logger.warning(
                            f"Failed to fetch sitemap {sitemap_url}: {response.status}"
                        )
                        return urls, nested_sitemaps

                    content = await response.text()

            # Parse XML content
            try:
                root = ET.fromstring(content)

                # Handle sitemap index files
                if root.tag.endswith("sitemapindex"):
                    for sitemap in root.findall(
                        ".//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap"
                    ):
                        loc_elem = sitemap.find(
                            "{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
                        )
                        lastmod_elem = sitemap.find(
                            "{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod"
                        )

                        if loc_elem is not None:
                            sitemap_info = SitemapInfo(
                                url=loc_elem.text,
                                last_modified=self._parse_date(
                                    lastmod_elem.text
                                    if lastmod_elem is not None
                                    else None
                                ),
                            )
                            nested_sitemaps.append(sitemap_info)

                # Handle regular sitemaps
                elif root.tag.endswith("urlset"):
                    for url_elem in root.findall(
                        ".//{http://www.sitemaps.org/schemas/sitemap/0.9}url"
                    ):
                        loc_elem = url_elem.find(
                            "{http://www.sitemaps.org/schemas/sitemap/0.9}loc"
                        )
                        if loc_elem is None:
                            continue

                        # Extract URL information
                        url = loc_elem.text
                        lastmod_elem = url_elem.find(
                            "{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod"
                        )
                        changefreq_elem = url_elem.find(
                            "{http://www.sitemaps.org/schemas/sitemap/0.9}changefreq"
                        )
                        priority_elem = url_elem.find(
                            "{http://www.sitemaps.org/schemas/sitemap/0.9}priority"
                        )

                        url_info = URLInfo(
                            url=url,
                            priority=(
                                float(priority_elem.text)
                                if priority_elem is not None
                                else 1.0
                            ),
                            last_modified=self._parse_date(
                                lastmod_elem.text if lastmod_elem is not None else None
                            ),
                            change_frequency=(
                                changefreq_elem.text
                                if changefreq_elem is not None
                                else None
                            ),
                            source="sitemap",
                        )
                        urls.append(url_info)

                logger.info(
                    f"Parsed sitemap {sitemap_url}: {len(urls)} URLs, {len(nested_sitemaps)} nested sitemaps"
                )

            except ET.ParseError as e:
                logger.error(f"Failed to parse XML from {sitemap_url}: {e}")

        except Exception as e:
            logger.error(f"Error fetching sitemap {sitemap_url}: {e}")

        return urls, nested_sitemaps

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO date string to datetime."""
        if not date_str:
            return None

        try:
            # Try different date formats
            for fmt in [
                "%Y-%m-%d",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S%z",
            ]:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
        except Exception:
            pass

        return None


class URLDiscoveryService:
    """Service for intelligent URL discovery and prioritization."""

    def __init__(self):
        self.sitemap_parser = SitemapParser()
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def discover_urls(
        self,
        base_url: str,
        keywords: Optional[List[str]] = None,
        max_urls: int = 1000,
        respect_robots: bool = True,
    ) -> List[URLInfo]:
        """Discover URLs from a base URL using sitemaps and other methods."""

        all_urls = []
        base_domain = URLUtils.get_base_url(base_url)

        # Step 1: Check robots.txt for sitemap URLs and crawling rules
        robots_parser = None
        if respect_robots:
            robots_parser = await self._fetch_robots_txt(base_domain)

        # Step 2: Discover sitemaps
        sitemap_urls = []
        if robots_parser and robots_parser.sitemaps:
            sitemap_urls.extend(robots_parser.sitemaps)
        else:
            # Try common sitemap locations
            common_sitemaps = [
                f"{base_domain}/sitemap.xml",
                f"{base_domain}/sitemap_index.xml",
                f"{base_domain}/sitemaps.xml",
            ]
            sitemap_urls.extend(common_sitemaps)

        # Step 3: Parse all discovered sitemaps
        for sitemap_url in sitemap_urls:
            urls = await self._parse_sitemap_recursive(sitemap_url, max_depth=3)

            # Filter by robots.txt rules
            if robots_parser:
                urls = [url for url in urls if robots_parser.is_allowed(url.url)]

            all_urls.extend(urls)

            if len(all_urls) >= max_urls:
                break

        # Step 4: Score URLs by relevance if keywords provided
        if keywords:
            all_urls = self._score_urls_by_keywords(all_urls, keywords)

        # Step 5: Sort by priority and relevance, limit results
        all_urls.sort(key=lambda x: (x.relevance_score, x.priority), reverse=True)

        logger.info(f"Discovered {len(all_urls[:max_urls])} URLs from {base_url}")
        return all_urls[:max_urls]

    async def _fetch_robots_txt(self, base_url: str) -> Optional[RobotsTxtParser]:
        """Fetch and parse robots.txt from base URL."""
        robots_url = f"{base_url}/robots.txt"

        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            async with self.session.get(robots_url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    parser = RobotsTxtParser(content)
                    logger.info(
                        f"Parsed robots.txt from {robots_url}: "
                        f"{len(parser.sitemaps)} sitemaps, "
                        f"{len(parser.disallowed_paths)} disallowed paths"
                    )
                    return parser
        except Exception as e:
            logger.debug(f"Could not fetch robots.txt from {robots_url}: {e}")

        return None

    async def _parse_sitemap_recursive(
        self, sitemap_url: str, max_depth: int = 3
    ) -> List[URLInfo]:
        """Recursively parse sitemap and nested sitemaps."""
        if max_depth <= 0:
            return []

        all_urls = []
        urls, nested_sitemaps = await self.sitemap_parser.parse_sitemap(sitemap_url)
        all_urls.extend(urls)

        # Parse nested sitemaps
        for sitemap_info in nested_sitemaps:
            nested_urls = await self._parse_sitemap_recursive(
                sitemap_info.url, max_depth - 1
            )
            all_urls.extend(nested_urls)

        return all_urls

    def _score_urls_by_keywords(
        self, urls: List[URLInfo], keywords: List[str]
    ) -> List[URLInfo]:
        """Score URLs based on keyword relevance in URL path."""
        keyword_patterns = [re.compile(re.escape(kw), re.IGNORECASE) for kw in keywords]

        for url_info in urls:
            score = 0.0
            url_lower = url_info.url.lower()

            # Score based on keyword matches in URL
            for pattern in keyword_patterns:
                if pattern.search(url_lower):
                    score += 1.0

            # Bonus for documentation-like paths
            doc_patterns = [
                r"/docs?/",
                r"/guide/",
                r"/tutorial/",
                r"/api/",
                r"/reference/",
                r"/help/",
            ]

            for pattern in doc_patterns:
                if re.search(pattern, url_lower):
                    score += 0.5

            # Normalize score
            url_info.relevance_score = (
                min(score / len(keywords), 2.0) if keywords else 1.0
            )

        return urls


__all__ = [
    "URLDiscoveryService",
    "URLInfo",
    "SitemapInfo",
    "RobotsTxtParser",
    "SitemapParser",
]
