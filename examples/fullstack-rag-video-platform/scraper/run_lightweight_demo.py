#!/usr/bin/env python3
"""
Lightweight Demo - No browser needed
Uses httpx to analyze DealMachine.com structure
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lightweight-demo")


async def run_lightweight_demo():
    """Run lightweight demo without browser"""
    print("\n" + "="*70)
    print("🔥 DEALMACHINE SENSEI - LIGHTWEIGHT DEMO")
    print("="*70 + "\n")

    print("⚡ Running lightweight mode (no browser needed)")
    print("📊 Will analyze DealMachine.com public pages\n")

    # Create documents directory
    docs_dir = Path("/home/user/Documents/dealmachine_data")
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Create HTTP client
    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
    ) as client:

        # Fetch homepage
        print("🌐 Fetching DealMachine.com homepage...")
        try:
            response = await client.get("https://www.dealmachine.com")
            print(f"✓ Status: {response.status_code}")

            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Get title
            title = soup.find('title')
            title_text = title.get_text() if title else "No title"
            print(f"✓ Page title: {title_text}")

            # Analyze structure
            print("\n🔍 Analyzing page structure...")

            # Count elements
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            links = soup.find_all('a')
            buttons = soup.find_all('button')
            forms = soup.find_all('form')
            images = soup.find_all('img')

            print(f"  • Headings: {len(headings)}")
            print(f"  • Links: {len(links)}")
            print(f"  • Buttons: {len(buttons)}")
            print(f"  • Forms: {len(forms)}")
            print(f"  • Images: {len(images)}")

            # Extract some content
            print("\n📝 Extracting content...")

            h1_tags = soup.find_all('h1')
            if h1_tags:
                print(f"  • Main heading: {h1_tags[0].get_text().strip()}")

            # Look for navigation
            nav = soup.find('nav')
            if nav:
                nav_links = nav.find_all('a')
                print(f"  • Navigation links: {len(nav_links)}")

            # Save analysis
            analysis = {
                "url": "https://www.dealmachine.com",
                "analyzed_at": datetime.utcnow().isoformat(),
                "status_code": response.status_code,
                "title": title_text,
                "stats": {
                    "headings": len(headings),
                    "links": len(links),
                    "buttons": len(buttons),
                    "forms": len(forms),
                    "images": len(images)
                },
                "sample_headings": [h.get_text().strip() for h in headings[:5]],
                "sample_links": [
                    {
                        "text": link.get_text().strip(),
                        "href": link.get('href', '')
                    }
                    for link in links[:10] if link.get_text().strip()
                ]
            }

            # Save to file
            analysis_file = docs_dir / "dealmachine_analysis.json"
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"\n💾 Analysis saved: {analysis_file}")

            # Save HTML for reference
            html_file = docs_dir / "dealmachine_homepage.html"
            with open(html_file, 'w') as f:
                f.write(response.text)
            print(f"💾 HTML saved: {html_file}")

            # Create a quick summary report
            report_lines = [
                "="*70,
                "DEALMACHINE.COM STRUCTURE ANALYSIS",
                "="*70,
                f"Analyzed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
                f"URL: https://www.dealmachine.com",
                f"Status: {response.status_code}",
                "",
                "PAGE STRUCTURE:",
                f"  • Title: {title_text}",
                f"  • Headings: {len(headings)}",
                f"  • Links: {len(links)}",
                f"  • Buttons: {len(buttons)}",
                f"  • Forms: {len(forms)}",
                f"  • Images: {len(images)}",
                "",
                "SAMPLE HEADINGS:",
            ]

            for h in headings[:5]:
                report_lines.append(f"  • {h.get_text().strip()}")

            report_lines.extend([
                "",
                "NAVIGATION LINKS:",
            ])

            for link in links[:10]:
                link_text = link.get_text().strip()
                if link_text:
                    report_lines.append(f"  • {link_text}")

            report_lines.extend([
                "",
                "="*70,
                "NOTES:",
                "• This is a public page analysis (no login)",
                "• For full property scraping, credentials are needed",
                "• The SENSEI learns patterns from both public and authenticated pages",
                "="*70
            ])

            report = "\n".join(report_lines)

            # Save report
            report_file = docs_dir / "analysis_report.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            print(f"💾 Report saved: {report_file}")

            # Print report
            print("\n" + report)

        except Exception as e:
            logger.error(f"Error fetching page: {e}")
            print(f"\n❌ Error: {e}")
            print("This might be due to network restrictions or rate limiting.")

    print("\n" + "="*70)
    print("✅ LIGHTWEIGHT DEMO COMPLETE!")
    print(f"📁 Results saved to: {docs_dir}")
    print("="*70)

    print("\n💡 Next Steps:")
    print("  1. Check the Documents folder for analysis results")
    print("  2. For full scraping with login, provide DealMachine credentials")
    print("  3. The SENSEI will learn patterns and improve over time")
    print("\n")


if __name__ == "__main__":
    asyncio.run(run_lightweight_demo())
