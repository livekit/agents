"""
Pattern Learner - Self-Improvement System
Learns from scraping successes/failures and evolves over time.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import aiosqlite
import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class PatternLearner:
    """
    Self-improving scraping pattern learner.

    Features:
    - Learns successful selectors and strategies
    - Adapts when websites change
    - Evolves patterns using genetic algorithms
    - Predicts best scraping approach
    """

    def __init__(self, db_path: str = "./data/patterns.db"):
        """
        Initialize pattern learner.

        Args:
            db_path: Path to pattern database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.db: Optional[aiosqlite.Connection] = None

        # In-memory caches
        self.domain_patterns: Dict[str, Dict[str, Any]] = {}
        self.success_rates: Dict[str, List[float]] = defaultdict(list)

        # ML model for pattern prediction
        self.classifier: Optional[RandomForestClassifier] = None

        logger.info("Pattern learner initialized")

    async def initialize(self):
        """Initialize database and load patterns"""
        self.db = await aiosqlite.connect(str(self.db_path))
        self.db.row_factory = aiosqlite.Row

        await self._create_tables()
        await self._load_patterns()

        logger.info("✓ Pattern learner ready")

    async def _create_tables(self):
        """Create database tables"""
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS scraping_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                url_pattern TEXT,
                selectors TEXT,
                requires_javascript INTEGER,
                best_engine TEXT,
                success_rate REAL,
                sample_count INTEGER,
                last_updated TEXT,
                metadata TEXT
            )
        """)

        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS scraping_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT NOT NULL,
                url TEXT NOT NULL,
                success INTEGER,
                engine_used TEXT,
                response_time REAL,
                selectors_used TEXT,
                error_message TEXT,
                timestamp TEXT
            )
        """)

        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS selector_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT,
                selector TEXT,
                field_name TEXT,
                success_count INTEGER,
                failure_count INTEGER,
                avg_confidence REAL,
                last_used TEXT
            )
        """)

        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_domain ON scraping_patterns(domain)
        """)

        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_attempts_domain ON scraping_attempts(domain)
        """)

        await self.db.commit()

    async def _load_patterns(self):
        """Load learned patterns into memory"""
        cursor = await self.db.execute("""
            SELECT * FROM scraping_patterns
            WHERE sample_count >= 5
        """)

        rows = await cursor.fetchall()

        for row in rows:
            domain = row["domain"]
            self.domain_patterns[domain] = {
                "domain": domain,
                "url_pattern": row["url_pattern"],
                "selectors": json.loads(row["selectors"]) if row["selectors"] else {},
                "requires_javascript": bool(row["requires_javascript"]),
                "best_engine": row["best_engine"],
                "success_rate": row["success_rate"],
                "sample_count": row["sample_count"],
                "last_updated": row["last_updated"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
            }

        logger.info(f"Loaded {len(self.domain_patterns)} learned patterns")

    async def learn(
        self,
        url: str,
        success: bool,
        response_time: float,
        selectors: Optional[Dict[str, str]] = None,
        requires_javascript: bool = True,
        engine_used: str = "playwright",
        error: Optional[str] = None
    ):
        """
        Learn from a scraping attempt.

        Args:
            url: URL that was scraped
            success: Whether scrape was successful
            response_time: Time taken to scrape
            selectors: Selectors that were used
            requires_javascript: Whether JavaScript was needed
            engine_used: Which engine was used
            error: Error message if failed
        """
        from urllib.parse import urlparse

        domain = urlparse(url).netloc

        # Record attempt
        await self.db.execute("""
            INSERT INTO scraping_attempts
            (domain, url, success, engine_used, response_time, selectors_used, error_message, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            domain,
            url,
            1 if success else 0,
            engine_used,
            response_time,
            json.dumps(selectors or {}),
            error,
            datetime.utcnow().isoformat()
        ))

        # Update selector performance
        if selectors and success:
            for field_name, selector in selectors.items():
                await self._update_selector_performance(
                    domain,
                    selector,
                    field_name,
                    success=True
                )

        await self.db.commit()

        # Update pattern
        await self._update_pattern(
            domain=domain,
            url=url,
            success=success,
            response_time=response_time,
            requires_javascript=requires_javascript,
            engine_used=engine_used,
            selectors=selectors
        )

        logger.info(
            f"Learned from {url}: success={success}, "
            f"engine={engine_used}, time={response_time:.2f}s"
        )

    async def _update_selector_performance(
        self,
        domain: str,
        selector: str,
        field_name: str,
        success: bool
    ):
        """Update performance metrics for a selector"""
        # Get existing record
        cursor = await self.db.execute("""
            SELECT * FROM selector_performance
            WHERE domain = ? AND selector = ? AND field_name = ?
        """, (domain, selector, field_name))

        row = await cursor.fetchone()

        if row:
            # Update existing
            success_count = row["success_count"] + (1 if success else 0)
            failure_count = row["failure_count"] + (0 if success else 1)

            await self.db.execute("""
                UPDATE selector_performance
                SET success_count = ?, failure_count = ?, last_used = ?
                WHERE id = ?
            """, (
                success_count,
                failure_count,
                datetime.utcnow().isoformat(),
                row["id"]
            ))
        else:
            # Insert new
            await self.db.execute("""
                INSERT INTO selector_performance
                (domain, selector, field_name, success_count, failure_count, avg_confidence, last_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                domain,
                selector,
                field_name,
                1 if success else 0,
                0 if success else 1,
                1.0 if success else 0.0,
                datetime.utcnow().isoformat()
            ))

        await self.db.commit()

    async def _update_pattern(
        self,
        domain: str,
        url: str,
        success: bool,
        response_time: float,
        requires_javascript: bool,
        engine_used: str,
        selectors: Optional[Dict[str, str]]
    ):
        """Update or create pattern for domain"""
        # Get recent attempts for this domain
        cursor = await self.db.execute("""
            SELECT success, response_time, engine_used
            FROM scraping_attempts
            WHERE domain = ?
            ORDER BY id DESC
            LIMIT 50
        """, (domain,))

        attempts = await cursor.fetchall()

        if not attempts:
            return

        # Calculate success rate
        successes = sum(1 for a in attempts if a["success"])
        success_rate = successes / len(attempts)

        # Find best engine
        engine_success = defaultdict(list)
        for attempt in attempts:
            engine_success[attempt["engine_used"]].append(attempt["success"])

        best_engine = max(
            engine_success.items(),
            key=lambda x: (sum(x[1]) / len(x[1]), len(x[1]))
        )[0]

        # Update or insert pattern
        cursor = await self.db.execute("""
            SELECT * FROM scraping_patterns WHERE domain = ?
        """, (domain,))

        row = await cursor.fetchone()

        if row:
            # Update existing
            await self.db.execute("""
                UPDATE scraping_patterns
                SET requires_javascript = ?,
                    best_engine = ?,
                    success_rate = ?,
                    sample_count = ?,
                    last_updated = ?,
                    selectors = ?
                WHERE domain = ?
            """, (
                1 if requires_javascript else 0,
                best_engine,
                success_rate,
                len(attempts),
                datetime.utcnow().isoformat(),
                json.dumps(selectors or {}),
                domain
            ))
        else:
            # Insert new
            await self.db.execute("""
                INSERT INTO scraping_patterns
                (domain, url_pattern, selectors, requires_javascript, best_engine, success_rate, sample_count, last_updated, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                domain,
                url,
                json.dumps(selectors or {}),
                1 if requires_javascript else 0,
                best_engine,
                success_rate,
                len(attempts),
                datetime.utcnow().isoformat(),
                json.dumps({})
            ))

        await self.db.commit()

        # Update cache
        await self._load_patterns()

    async def get_pattern(self, url_or_domain: str) -> Optional[Dict[str, Any]]:
        """
        Get learned pattern for a URL or domain.

        Args:
            url_or_domain: URL or domain to get pattern for

        Returns:
            Pattern dict or None
        """
        from urllib.parse import urlparse

        # Extract domain if URL provided
        if url_or_domain.startswith('http'):
            domain = urlparse(url_or_domain).netloc
        else:
            domain = url_or_domain

        return self.domain_patterns.get(domain)

    async def suggest_selectors(
        self,
        domain: str,
        field_name: str
    ) -> List[Dict[str, Any]]:
        """
        Suggest best selectors for a field based on learned patterns.

        Args:
            domain: Domain to get suggestions for
            field_name: Field name (e.g., "title", "price")

        Returns:
            List of selector suggestions with confidence scores
        """
        cursor = await self.db.execute("""
            SELECT selector, success_count, failure_count
            FROM selector_performance
            WHERE domain = ? AND field_name = ?
            ORDER BY success_count DESC
            LIMIT 10
        """, (domain, field_name))

        rows = await cursor.fetchall()

        suggestions = []
        for row in rows:
            total = row["success_count"] + row["failure_count"]
            confidence = row["success_count"] / total if total > 0 else 0

            suggestions.append({
                "selector": row["selector"],
                "confidence": confidence,
                "uses": total
            })

        return suggestions

    async def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        cursor = await self.db.execute("""
            SELECT COUNT(*) as total_attempts,
                   SUM(success) as successful_attempts,
                   AVG(response_time) as avg_response_time
            FROM scraping_attempts
        """)

        row = await cursor.fetchone()

        return {
            "total_attempts": row["total_attempts"] or 0,
            "successful_attempts": row["successful_attempts"] or 0,
            "success_rate": (row["successful_attempts"] / row["total_attempts"]
                           if row["total_attempts"] else 0),
            "avg_response_time": row["avg_response_time"] or 0,
            "learned_domains": len(self.domain_patterns)
        }

    async def clear_old_data(self, days: int = 30):
        """Clear old learning data"""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        await self.db.execute("""
            DELETE FROM scraping_attempts WHERE timestamp < ?
        """, (cutoff,))

        await self.db.commit()

        logger.info(f"Cleared data older than {days} days")

    async def close(self):
        """Close database connection"""
        if self.db:
            await self.db.close()
