# 🚀 BEAST Scraper - Quick Start Guide

Get scraping in under 5 minutes!

## 📦 Installation

### 1. Install Dependencies

```bash
cd scraper/

# Install Python packages
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

### 2. Install Open-Source LLM (Ollama)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (choose one)
ollama pull llama3.2  # Recommended: fast and capable
# or
ollama pull mistral   # Alternative: also excellent

# Verify it's running
ollama list
```

### 3. Install Whisper (for voice)

```bash
# Already in requirements.txt, but verify:
python -c "import whisper; print(whisper.__version__)"

# Download model (optional, happens automatically on first use)
python -c "import whisper; whisper.load_model('base')"
```

## 🎯 Basic Usage

### Example 1: Simple Scraping

```python
import asyncio
from beast_scraper import scrape

async def main():
    # Scrape a single URL
    result = await scrape("https://news.ycombinator.com")

    if result.success:
        print(f"Scraped {result.url} in {result.response_time:.2f}s")
        print(f"Data: {result.data}")

asyncio.run(main())
```

### Example 2: Extract Specific Fields

```python
from beast_scraper import scrape

async def main():
    # Extract specific data
    result = await scrape(
        "https://example.com/products",
        extract=["title", "price", "description"],
        mode="browser"  # Use full browser for JS sites
    )

    print(result.data)

asyncio.run(main())
```

### Example 3: Scrape Multiple URLs

```python
from beast_scraper import scrape_many

async def main():
    urls = [
        "https://news.ycombinator.com",
        "https://techcrunch.com",
        "https://theverge.com"
    ]

    # Scrape all in parallel!
    results = await scrape_many(urls, max_concurrent=10)

    for result in results:
        if result.success:
            print(f"✓ {result.url}: {len(result.data)} items")
        else:
            print(f"✗ {result.url}: {result.error}")

asyncio.run(main())
```

## 🗣️ Conversational Scraping

Talk to the scraper in natural language!

```python
from conversational_scraper import ConversationalScraper

async def main():
    async with ConversationalScraper(user_id="your_name") as scraper:
        # Natural language request
        response = await scraper.chat(
            "Scrape the top 10 articles from Hacker News and give me the titles"
        )
        print(response)

        # Follow-up (it remembers context!)
        response = await scraper.chat(
            "Now do the same for TechCrunch"
        )
        print(response)

        # Complex multi-step
        response = await scraper.chat(
            "Compare prices for 'wireless mouse' on Amazon and Best Buy"
        )
        print(response)

asyncio.run(main())
```

## 🔐 Auto-Login Example

```python
from beast_scraper import BeastScraper

async def main():
    async with BeastScraper() as scraper:
        # First, add credentials to vault
        await scraper.credential_vault.add_credentials(
            site="example.com",
            username="your_username",
            password="your_password"
        )

        # Now scrape with auto-login
        result = await scraper.scrape(
            "https://example.com/members-only",
            login=True
        )

        # Session is maintained!
        result2 = await scraper.scrape(
            "https://example.com/profile"
        )

asyncio.run(main())
```

## 🧠 Self-Improvement in Action

The scraper learns from every scrape:

```python
from beast_scraper import BeastScraper

async def main():
    async with BeastScraper() as scraper:
        # First scrape - learns patterns
        result1 = await scraper.scrape("https://news.ycombinator.com")
        print(f"First scrape: {result1.response_time:.2f}s")

        # Second scrape - uses learned patterns (faster!)
        result2 = await scraper.scrape("https://news.ycombinator.com")
        print(f"Second scrape: {result2.response_time:.2f}s (improved!)")

        # View learned patterns
        pattern = await scraper.pattern_learner.get_pattern("news.ycombinator.com")
        print(f"Learned: {pattern}")

asyncio.run(main())
```

## 🎮 Advanced Examples

### Monitor Prices

```python
async def price_monitor():
    async with BeastScraper() as scraper:
        products = {
            "amazon": "https://amazon.com/dp/PRODUCT_ID",
            "bestbuy": "https://bestbuy.com/product/ID"
        }

        while True:
            for site, url in products.items():
                result = await scraper.scrape(url, extract=["price"])
                print(f"{site}: ${result.data.get('price')}")

            await asyncio.sleep(3600)  # Check every hour

asyncio.run(price_monitor())
```

### News Aggregation

```python
async def aggregate_news():
    sources = [
        "https://news.ycombinator.com",
        "https://techcrunch.com",
        "https://arstechnica.com"
    ]

    results = await scrape_many(
        sources,
        extract=["title", "url", "summary"],
        mode="auto"
    )

    for result in results:
        print(f"\nFrom {result.url}:")
        for article in result.data.get('articles', []):
            print(f"  - {article['title']}")

asyncio.run(aggregate_news())
```

### E-commerce Research

```python
async def product_research():
    async with ConversationalScraper(user_id="researcher") as scraper:
        response = await scraper.chat("""
            Research 'mechanical keyboard' on Amazon:
            1. Find top 10 products
            2. Get prices and ratings
            3. Extract pros/cons from reviews
            4. Give me a summary and recommendation
        """)

        print(response)

asyncio.run(product_research())
```

## ⚙️ Configuration

Create `scraper_config.yaml`:

```yaml
# Engines
primary_engine: playwright
fallback_engine: httpx

# Performance
concurrency:
  max_workers: 10
  timeout: 30

# Rate Limiting
rate_limiting:
  requests_per_second: 5
  respect_robots_txt: true

# LLM (Ollama)
llm:
  provider: ollama
  model: llama3.2
  base_url: http://localhost:11434

# Learning
learning:
  enabled: true
  pattern_db: ./data/patterns.db
```

## 🔧 CLI Usage

```bash
# Simple scrape
python -m scraper scrape https://example.com

# Extract specific fields
python -m scraper scrape https://example.com --extract title,price

# Multiple URLs
python -m scraper scrape-many urls.txt --output results.json

# Conversational mode
python -m scraper chat --user john_doe
```

## 🎤 Voice Control (LiveKit)

```python
from voice_scraper import VoiceScraper

async def main():
    # Start voice interface
    scraper = VoiceScraper(
        livekit_url="ws://localhost:7880",
        user_id="voice_user"
    )

    await scraper.start()

    # Now speak:
    # "Scrape the top tech news from TechCrunch"
    # Bot will respond via voice and execute!

asyncio.run(main())
```

## 📊 View Statistics

```python
async def view_stats():
    async with BeastScraper() as scraper:
        stats = await scraper.pattern_learner.get_statistics()

        print(f"Total scrapes: {stats['total_attempts']}")
        print(f"Success rate: {stats['success_rate']:.1%}")
        print(f"Avg response time: {stats['avg_response_time']:.2f}s")
        print(f"Learned domains: {stats['learned_domains']}")

asyncio.run(view_stats())
```

## 🐛 Debugging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Or configure per module
logging.getLogger("beast_scraper").setLevel(logging.DEBUG)
```

## 🚦 Best Practices

1. **Start with `mode="auto"`** - Let the scraper choose the best engine
2. **Use `mode="fast"`** for API endpoints and static sites
3. **Use `mode="browser"`** for JavaScript-heavy sites
4. **Enable learning** - `learn=True` to improve over time
5. **Respect robots.txt** - Be a good internet citizen
6. **Use rate limiting** - Don't overwhelm servers
7. **Cache results** - Enable caching for frequently scraped URLs
8. **Monitor performance** - Check learning stats regularly

## 🆘 Troubleshooting

### Ollama not responding
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Playwright browser issues
```bash
# Reinstall browsers
playwright install --force chromium
```

### Slow scraping
```python
# Use fast mode for simple sites
result = await scrape(url, mode="fast")

# Or increase concurrency
results = await scrape_many(urls, max_concurrent=20)
```

### Memory issues with many URLs
```python
# Process in batches
async def scrape_in_batches(urls, batch_size=100):
    for i in range(0, len(urls), batch_size):
        batch = urls[i:i+batch_size]
        results = await scrape_many(batch)
        # Process results
        yield results
```

## 📚 Next Steps

1. **Read the full docs**: `SCRAPING_BEAST.md`
2. **Explore examples**: Check `examples/` directory
3. **Try MCP integration**: `mcp_examples.py`
4. **Build custom extractors**: `custom_extractor_guide.md`
5. **Deploy to production**: `deployment_guide.md`

## 💡 Pro Tips

- **Combine with RAG**: Scrape data and add to knowledge base automatically
- **Schedule scrapes**: Use cron or Celery for periodic scraping
- **Use webhooks**: Get notified when scraped data changes
- **Build pipelines**: Chain multiple scrapes together
- **Export data**: Save to CSV, JSON, or database
- **Visualize results**: Create dashboards with scraped data

---

**Happy scraping! 🦾**

Questions? Check the docs or open an issue!
