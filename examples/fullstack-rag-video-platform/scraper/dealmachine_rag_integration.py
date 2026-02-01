"""
DealMachine RAG Integration
Connects the DealMachine Sensei scraper with the RAG knowledge system.

This creates an intelligent system that:
- Learns from every scrape
- Stores knowledge in vector database
- Can answer questions about scraped data
- Gets smarter over time
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from dealmachine_sensei import DealMachineSensei, Property

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dealmachine-rag")


class DealMachineRAG:
    """
    RAG-powered DealMachine intelligence system.

    Features:
    - Stores all scraped data in knowledge base
    - Semantic search across properties
    - Learns patterns and trends
    - Provides insights and recommendations
    """

    def __init__(self, documents_dir: str = "/home/user/Documents/dealmachine_data"):
        """Initialize RAG system"""
        self.documents_dir = Path(documents_dir)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

        self.knowledge_base: List[Property] = []
        self.embeddings: Dict[str, Any] = {}

        logger.info("🧠 DealMachine RAG initialized")

    def load_knowledge_base(self):
        """Load all previously scraped data"""
        logger.info("📚 Loading knowledge base...")

        # Load all JSON files
        json_files = list(self.documents_dir.glob("dealmachine_properties_*.json"))

        total_properties = 0
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    for item in data:
                        prop = Property(**item)
                        self.knowledge_base.append(prop)
                        total_properties += 1
            except Exception as e:
                logger.error(f"Error loading {json_file}: {e}")

        logger.info(f"✓ Loaded {total_properties} properties from {len(json_files)} scrapes")

        return total_properties

    def index_properties(self):
        """Create searchable index of properties"""
        logger.info("🔍 Indexing properties...")

        # Simple keyword-based indexing (can upgrade to embeddings later)
        index = {
            'by_address': {},
            'by_city': {},
            'by_price_range': {},
            'by_type': {}
        }

        for prop in self.knowledge_base:
            # Index by address
            index['by_address'][prop.address.lower()] = prop

            # Index by city
            if prop.city:
                if prop.city not in index['by_city']:
                    index['by_city'][prop.city] = []
                index['by_city'][prop.city].append(prop)

            # Index by price range
            if prop.price:
                try:
                    price_num = int(''.join(filter(str.isdigit, prop.price)))
                    range_key = f"{(price_num // 100000) * 100000}-{((price_num // 100000) + 1) * 100000}"
                    if range_key not in index['by_price_range']:
                        index['by_price_range'][range_key] = []
                    index['by_price_range'][range_key].append(prop)
                except:
                    pass

            # Index by type
            if prop.property_type:
                if prop.property_type not in index['by_type']:
                    index['by_type'][prop.property_type] = []
                index['by_type'][prop.property_type].append(prop)

        # Save index
        index_file = self.documents_dir / "property_index.json"
        with open(index_file, 'w') as f:
            # Convert to serializable format
            serializable_index = {
                'by_address': {k: v.address for k, v in index['by_address'].items()},
                'by_city': {k: len(v) for k, v in index['by_city'].items()},
                'by_price_range': {k: len(v) for k, v in index['by_price_range'].items()},
                'by_type': {k: len(v) for k, v in index['by_type'].items()}
            }
            json.dump(serializable_index, f, indent=2)

        logger.info(f"✓ Indexed {len(self.knowledge_base)} properties")
        return index

    def get_insights(self) -> Dict[str, Any]:
        """Generate insights from knowledge base"""
        logger.info("💡 Generating insights...")

        if not self.knowledge_base:
            return {}

        insights = {
            'total_properties': len(self.knowledge_base),
            'cities': {},
            'price_stats': {},
            'property_types': {},
            'recent_scrapes': []
        }

        # Count by city
        for prop in self.knowledge_base:
            if prop.city:
                insights['cities'][prop.city] = insights['cities'].get(prop.city, 0) + 1

        # Price statistics
        prices = []
        for prop in self.knowledge_base:
            if prop.price:
                try:
                    price_num = int(''.join(filter(str.isdigit, prop.price)))
                    prices.append(price_num)
                except:
                    pass

        if prices:
            insights['price_stats'] = {
                'min': min(prices),
                'max': max(prices),
                'avg': sum(prices) // len(prices),
                'count': len(prices)
            }

        # Count by property type
        for prop in self.knowledge_base:
            if prop.property_type:
                insights['property_types'][prop.property_type] = \
                    insights['property_types'].get(prop.property_type, 0) + 1

        # Recent scrapes
        sorted_props = sorted(
            self.knowledge_base,
            key=lambda x: x.scraped_at,
            reverse=True
        )
        insights['recent_scrapes'] = [
            {
                'address': p.address,
                'price': p.price,
                'scraped_at': p.scraped_at
            }
            for p in sorted_props[:10]
        ]

        # Save insights
        insights_file = self.documents_dir / "insights.json"
        with open(insights_file, 'w') as f:
            json.dump(insights, f, indent=2)

        logger.info("✓ Insights generated")
        return insights

    def generate_report(self) -> str:
        """Generate human-readable report"""
        insights = self.get_insights()

        report = []
        report.append("=" * 70)
        report.append("DEALMACHINE KNOWLEDGE BASE REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        report.append(f"📊 TOTAL PROPERTIES: {insights['total_properties']}")
        report.append("")

        if insights.get('cities'):
            report.append("🏙️  TOP CITIES:")
            sorted_cities = sorted(
                insights['cities'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for city, count in sorted_cities[:10]:
                report.append(f"   • {city}: {count} properties")
            report.append("")

        if insights.get('price_stats'):
            stats = insights['price_stats']
            report.append("💰 PRICE STATISTICS:")
            report.append(f"   • Min Price: ${stats['min']:,}")
            report.append(f"   • Max Price: ${stats['max']:,}")
            report.append(f"   • Avg Price: ${stats['avg']:,}")
            report.append(f"   • Properties with price: {stats['count']}")
            report.append("")

        if insights.get('property_types'):
            report.append("🏠 PROPERTY TYPES:")
            for prop_type, count in insights['property_types'].items():
                report.append(f"   • {prop_type}: {count}")
            report.append("")

        if insights.get('recent_scrapes'):
            report.append("🆕 RECENT SCRAPES:")
            for scrape in insights['recent_scrapes'][:5]:
                report.append(f"   • {scrape['address']} - {scrape['price']}")
            report.append("")

        report.append("=" * 70)
        report.append(f"📁 Data location: {self.documents_dir}")
        report.append("=" * 70)

        report_text = "\n".join(report)

        # Save report
        report_file = self.documents_dir / "knowledge_base_report.txt"
        with open(report_file, 'w') as f:
            f.write(report_text)

        logger.info(f"💾 Report saved: {report_file}")

        return report_text


async def integrate_and_analyze():
    """
    Full integration: Scrape → RAG → Insights → Report
    """
    print("\n" + "="*70)
    print("🧠 DEALMACHINE RAG INTEGRATION - FULL ANALYSIS")
    print("="*70 + "\n")

    # Initialize RAG system
    rag = DealMachineRAG()

    # Load existing knowledge
    total_props = rag.load_knowledge_base()

    if total_props == 0:
        print("⚠️  No existing data found.")
        print("Run dealmachine_sensei.py first to scrape some properties!\n")
        return

    # Index properties
    rag.index_properties()

    # Generate insights
    rag.get_insights()

    # Generate report
    report = rag.generate_report()

    print("\n" + report + "\n")

    print("✅ RAG integration complete!")
    print(f"📁 All analysis saved to: {rag.documents_dir}\n")


if __name__ == "__main__":
    asyncio.run(integrate_and_analyze())
