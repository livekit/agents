"""
BEAST Scraper - Open Source Web Scraping System
Fast. Intelligent. Self-Improving. Conversational.
"""

from beast_scraper import BeastScraper, scrape, scrape_many, ScrapeResult
from conversational_scraper import ConversationalScraper
from scraper_config import ScraperConfig

__version__ = "1.0.0"
__all__ = [
    "BeastScraper",
    "scrape",
    "scrape_many",
    "ConversationalScraper",
    "ScraperConfig",
    "ScrapeResult"
]
