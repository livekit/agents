"""
BEAST Scraper - Core Engine
The fastest, most intelligent open-source web scraper.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time

from playwright.async_api import async_playwright, Browser, Page
from bs4 import BeautifulSoup
import httpx
from selectolax.parser import HTMLParser

from scraper_config import ScraperConfig
from engines.playwright_engine import PlaywrightEngine
from engines.httpx_engine import HttpxEngine
from engines.scrapy_engine import ScrapyEngine
from learning.pattern_learner import PatternLearner
from auth.credential_vault import CredentialVault
from extractors.smart_extractor import SmartExtractor
from utils.performance import PerformanceMonitor

logger = logging.getLogger(__name__)


class ScraperMode(Enum):
    """Scraping modes"""
    AUTO = "auto"  # Auto-select best engine
    FAST = "fast"  # Httpx only (fastest)
    BROWSER = "browser"  # Playwright (full browser)
    COMPLEX = "complex"  # Scrapy (multi-page)
    STEALTH = "stealth"  # Anti-detection mode


@dataclass
class ScrapeRequest:
    """Scraping request configuration"""
    url: str
    mode: ScraperMode = ScraperMode.AUTO
    extract: Optional[List[str]] = None  # What to extract
    login: bool = False
    credentials: Optional[Dict[str, str]] = None
    wait_for: Optional[str] = None  # CSS selector to wait for
    javascript: bool = True  # Render JavaScript
    proxy: Optional[str] = None
    headers: Optional[Dict[str, str]] = None
    cookies: Optional[Dict[str, str]] = None
    timeout: int = 30
    retries: int = 3
    cache: bool = True
    learn: bool = True  # Learn from this scrape


@dataclass
class ScrapeResult:
    """Scraping result"""
    url: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    response_time: float = 0.0
    engine_used: str = ""
    learned_patterns: List[Dict[str, Any]] = None


class BeastScraper:
    """
    The BEAST - Blazingly fast, intelligent web scraper.

    Features:
    - Multi-engine (Playwright, Httpx, Scrapy)
    - Auto-login and session management
    - Self-improving pattern learning
    - API detection and direct calls
    - Parallel scraping with async
    """

    def __init__(self, config: Optional[ScraperConfig] = None):
        """Initialize the BEAST scraper"""
        self.config = config or ScraperConfig()

        # Engines
        self.playwright_engine: Optional[PlaywrightEngine] = None
        self.httpx_engine: Optional[HttpxEngine] = None
        self.scrapy_engine: Optional[ScrapyEngine] = None

        # Components
        self.pattern_learner = PatternLearner(db_path=self.config.learning.pattern_db)
        self.credential_vault = CredentialVault(
            vault_path=self.config.credentials.vault_path
        )
        self.extractor = SmartExtractor(self.pattern_learner)
        self.performance = PerformanceMonitor()

        # State
        self.initialized = False
        self._browser: Optional[Browser] = None

        logger.info("🦾 BEAST Scraper initialized")

    async def initialize(self):
        """Initialize scraping engines"""
        if self.initialized:
            return

        logger.info("Initializing scraping engines...")

        # Initialize engines
        self.httpx_engine = HttpxEngine(self.config)
        await self.httpx_engine.initialize()

        self.playwright_engine = PlaywrightEngine(self.config)
        await self.playwright_engine.initialize()

        # Pattern learner
        await self.pattern_learner.initialize()

        self.initialized = True
        logger.info("✓ All engines ready")

    async def scrape(
        self,
        url: str,
        mode: Union[ScraperMode, str] = ScraperMode.AUTO,
        **kwargs
    ) -> ScrapeResult:
        """
        Scrape a URL with intelligent engine selection.

        Args:
            url: URL to scrape
            mode: Scraping mode (auto, fast, browser, complex, stealth)
            **kwargs: Additional options (extract, login, etc.)

        Returns:
            ScrapeResult with extracted data
        """
        if not self.initialized:
            await self.initialize()

        # Convert string mode to enum
        if isinstance(mode, str):
            mode = ScraperMode(mode)

        # Build request
        request = ScrapeRequest(url=url, mode=mode, **kwargs)

        # Start performance tracking
        start_time = time.time()
        self.performance.start_request(url)

        try:
            # Auto-detect best mode
            if mode == ScraperMode.AUTO:
                mode = await self._select_best_mode(request)
                request.mode = mode

            # Handle login if needed
            if request.login:
                await self._handle_login(request)

            # Execute scrape based on mode
            if mode == ScraperMode.FAST:
                result = await self._scrape_fast(request)
            elif mode == ScraperMode.BROWSER:
                result = await self._scrape_browser(request)
            elif mode == ScraperMode.COMPLEX:
                result = await self._scrape_complex(request)
            elif mode == ScraperMode.STEALTH:
                result = await self._scrape_stealth(request)
            else:
                # Default to browser mode
                result = await self._scrape_browser(request)

            # Extract data using smart extractor
            if request.extract:
                result.data = await self.extractor.extract(
                    html=result.metadata.get('html', ''),
                    fields=request.extract,
                    url=url
                )

            # Learn from successful scrape
            if request.learn and result.success:
                await self._learn_from_scrape(request, result)

            # Update performance metrics
            result.response_time = time.time() - start_time
            self.performance.record_success(url, result.response_time)

            logger.info(
                f"✓ Scraped {url} in {result.response_time:.2f}s "
                f"using {result.engine_used}"
            )

            return result

        except Exception as e:
            logger.error(f"✗ Scrape failed for {url}: {e}")
            self.performance.record_failure(url)

            return ScrapeResult(
                url=url,
                data={},
                metadata={},
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )

    async def _select_best_mode(self, request: ScrapeRequest) -> ScraperMode:
        """
        Intelligently select the best scraping mode.

        Uses:
        - API detection (fastest)
        - Historical performance data
        - URL pattern analysis
        - Content type prediction
        """
        url = request.url

        # Check if API endpoint
        if await self._is_api_endpoint(url):
            logger.info(f"API detected for {url}, using FAST mode")
            return ScraperMode.FAST

        # Check learned patterns
        pattern = await self.pattern_learner.get_pattern(url)
        if pattern:
            if pattern.get('requires_javascript'):
                return ScraperMode.BROWSER
            elif pattern.get('is_static'):
                return ScraperMode.FAST

        # Check URL characteristics
        if any(hint in url.lower() for hint in ['/api/', '/v1/', '.json', '/graphql']):
            return ScraperMode.FAST

        # Default to browser for safety
        return ScraperMode.BROWSER

    async def _is_api_endpoint(self, url: str) -> bool:
        """Detect if URL is an API endpoint"""
        try:
            # Quick HEAD request
            async with httpx.AsyncClient() as client:
                response = await client.head(url, follow_redirects=True, timeout=5)
                content_type = response.headers.get('content-type', '')

                # Check if JSON/XML API
                if any(t in content_type for t in ['application/json', 'application/xml']):
                    return True
        except:
            pass

        return False

    async def _scrape_fast(self, request: ScrapeRequest) -> ScrapeResult:
        """Fast scraping using httpx (no browser)"""
        logger.info(f"Using FAST mode (httpx) for {request.url}")

        result = await self.httpx_engine.scrape(request)
        result.engine_used = "httpx"

        return result

    async def _scrape_browser(self, request: ScrapeRequest) -> ScrapeResult:
        """Browser-based scraping using Playwright"""
        logger.info(f"Using BROWSER mode (Playwright) for {request.url}")

        result = await self.playwright_engine.scrape(request)
        result.engine_used = "playwright"

        return result

    async def _scrape_complex(self, request: ScrapeRequest) -> ScrapeResult:
        """Complex multi-page scraping using Scrapy"""
        logger.info(f"Using COMPLEX mode (Scrapy) for {request.url}")

        # Scrapy integration (would launch Scrapy spider)
        # For now, fallback to browser
        return await self._scrape_browser(request)

    async def _scrape_stealth(self, request: ScrapeRequest) -> ScrapeResult:
        """Stealth mode with anti-detection"""
        logger.info(f"Using STEALTH mode for {request.url}")

        # Use playwright with stealth plugins
        result = await self.playwright_engine.scrape_stealth(request)
        result.engine_used = "playwright-stealth"

        return result

    async def _handle_login(self, request: ScrapeRequest):
        """Handle auto-login"""
        from auth.auto_login import AutoLogin

        logger.info(f"Attempting auto-login for {request.url}")

        auto_login = AutoLogin(
            credential_vault=self.credential_vault,
            playwright_engine=self.playwright_engine
        )

        # Get or use provided credentials
        if request.credentials:
            await auto_login.login(request.url, request.credentials)
        else:
            # Try to get from vault
            domain = self._extract_domain(request.url)
            credentials = await self.credential_vault.get_credentials(domain)

            if credentials:
                await auto_login.login(request.url, credentials)
            else:
                logger.warning(f"No credentials found for {domain}")

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc

    async def _learn_from_scrape(
        self,
        request: ScrapeRequest,
        result: ScrapeResult
    ):
        """Learn patterns from successful scrape"""
        await self.pattern_learner.learn(
            url=request.url,
            success=result.success,
            response_time=result.response_time,
            selectors=result.metadata.get('selectors', {}),
            requires_javascript=request.javascript,
            engine_used=result.engine_used
        )

    async def scrape_many(
        self,
        urls: List[str],
        mode: ScraperMode = ScraperMode.AUTO,
        max_concurrent: int = 10,
        **kwargs
    ) -> List[ScrapeResult]:
        """
        Scrape multiple URLs in parallel.

        Args:
            urls: List of URLs to scrape
            mode: Scraping mode
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional scrape options

        Returns:
            List of ScrapeResults
        """
        logger.info(f"Scraping {len(urls)} URLs with concurrency={max_concurrent}")

        # Create semaphore for rate limiting
        semaphore = asyncio.Semaphore(max_concurrent)

        async def scrape_with_limit(url: str):
            async with semaphore:
                return await self.scrape(url, mode=mode, **kwargs)

        # Execute all scrapes
        results = await asyncio.gather(
            *[scrape_with_limit(url) for url in urls],
            return_exceptions=True
        )

        # Filter exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error scraping {urls[i]}: {result}")
                valid_results.append(ScrapeResult(
                    url=urls[i],
                    data={},
                    metadata={},
                    success=False,
                    error=str(result)
                ))
            else:
                valid_results.append(result)

        success_count = sum(1 for r in valid_results if r.success)
        logger.info(f"✓ Completed {success_count}/{len(urls)} scrapes successfully")

        return valid_results

    async def close(self):
        """Clean up resources"""
        logger.info("Shutting down BEAST scraper...")

        if self.playwright_engine:
            await self.playwright_engine.close()

        if self.httpx_engine:
            await self.httpx_engine.close()

        logger.info("✓ Shutdown complete")

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


# Convenience functions
async def scrape(url: str, **kwargs) -> ScrapeResult:
    """Quick scrape function"""
    async with BeastScraper() as scraper:
        return await scraper.scrape(url, **kwargs)


async def scrape_many(urls: List[str], **kwargs) -> List[ScrapeResult]:
    """Quick multi-URL scrape"""
    async with BeastScraper() as scraper:
        return await scraper.scrape_many(urls, **kwargs)
