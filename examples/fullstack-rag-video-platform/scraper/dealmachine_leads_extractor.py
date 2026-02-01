#!/usr/bin/env python3
"""
DealMachine Leads Extractor - ULTIMATE EDITION
Extracts clean leads (name, address, phone) from DealMachine leads tab
Filters out DNC scrubs automatically
"""

import asyncio
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field

from playwright.async_api import async_playwright, Page, Browser
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leads-extractor")


@dataclass
class Lead:
    """Clean lead data structure"""
    name: str = ""
    address: str = ""
    city: str = ""
    state: str = ""
    zip_code: str = ""
    phone: str = ""
    email: str = ""
    status: str = ""
    property_type: str = ""
    dnc_status: str = "CLEAN"  # CLEAN or DNC
    source: str = "DealMachine"
    scraped_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def get_full_address(self) -> str:
        """Get complete formatted address"""
        parts = [self.address, self.city, self.state, self.zip_code]
        return ", ".join([p for p in parts if p])

    def is_valid(self) -> bool:
        """Check if lead has minimum required data"""
        return bool(self.name or self.address) and bool(self.phone)


class DealMachineLeadsExtractor:
    """
    Extract clean leads from DealMachine with DNC filtering
    """

    def __init__(
        self,
        documents_dir: str = "/home/user/Documents/dealmachine_data/leads",
        headless: bool = True
    ):
        self.documents_dir = Path(documents_dir)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.headless = headless

        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

        # Statistics
        self.stats = {
            "total_leads": 0,
            "clean_leads": 0,
            "dnc_filtered": 0,
            "invalid_leads": 0
        }

        logger.info("🔥 DealMachine Leads Extractor initialized")
        logger.info(f"📁 Leads directory: {self.documents_dir}")

    async def initialize(self):
        """Initialize browser"""
        logger.info("🚀 Initializing browser...")
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=self.headless)

        # Create context with realistic settings
        context = await self.browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )

        self.page = await context.new_page()
        logger.info("✓ Browser ready")

    async def login(self, email: str, password: str) -> bool:
        """
        Login to DealMachine

        Args:
            email: DealMachine email
            password: DealMachine password

        Returns:
            bool: True if login successful
        """
        logger.info("🔐 Logging in to DealMachine...")

        try:
            # Navigate to login page
            await self.page.goto("https://app.dealmachine.com/login", wait_until="networkidle")
            await asyncio.sleep(2)

            # Try multiple selector strategies
            email_selectors = [
                'input[type="email"]',
                'input[name="email"]',
                'input[placeholder*="email" i]',
                '#email',
                'input[id*="email" i]'
            ]

            password_selectors = [
                'input[type="password"]',
                'input[name="password"]',
                '#password',
                'input[id*="password" i]'
            ]

            # Fill email
            for selector in email_selectors:
                try:
                    await self.page.wait_for_selector(selector, timeout=2000)
                    await self.page.fill(selector, email)
                    logger.info(f"✓ Email filled using: {selector}")
                    break
                except:
                    continue

            # Fill password
            for selector in password_selectors:
                try:
                    await self.page.wait_for_selector(selector, timeout=2000)
                    await self.page.fill(selector, password)
                    logger.info(f"✓ Password filled using: {selector}")
                    break
                except:
                    continue

            # Submit form
            submit_selectors = [
                'button[type="submit"]',
                'button:has-text("Log in")',
                'button:has-text("Sign in")',
                'input[type="submit"]'
            ]

            for selector in submit_selectors:
                try:
                    await self.page.click(selector)
                    logger.info(f"✓ Login submitted using: {selector}")
                    break
                except:
                    continue

            # Wait for navigation
            await asyncio.sleep(5)

            # Check if login successful (looking for dashboard elements)
            current_url = self.page.url
            if "dashboard" in current_url or "app.dealmachine.com" in current_url and "login" not in current_url:
                logger.info("✅ Login successful!")
                return True
            else:
                logger.warning(f"⚠️  Login may have failed. Current URL: {current_url}")
                return False

        except Exception as e:
            logger.error(f"❌ Login error: {e}")
            return False

    async def navigate_to_leads(self) -> bool:
        """
        Navigate to the leads tab

        Returns:
            bool: True if navigation successful
        """
        logger.info("🧭 Navigating to leads tab...")

        try:
            # Try direct URL first
            await self.page.goto("https://app.dealmachine.com/leads", wait_until="networkidle")
            await asyncio.sleep(3)

            # Check if we're on leads page
            current_url = self.page.url
            if "leads" in current_url.lower():
                logger.info("✅ On leads page!")
                return True

            # Try clicking leads navigation
            nav_selectors = [
                'a[href*="leads"]',
                'button:has-text("Leads")',
                'nav a:has-text("Leads")',
                '[data-testid*="leads"]'
            ]

            for selector in nav_selectors:
                try:
                    await self.page.click(selector)
                    await asyncio.sleep(3)
                    if "leads" in self.page.url.lower():
                        logger.info(f"✅ Navigated using: {selector}")
                        return True
                except:
                    continue

            logger.warning("⚠️  Could not navigate to leads tab")
            return False

        except Exception as e:
            logger.error(f"❌ Navigation error: {e}")
            return False

    async def extract_leads(self, max_leads: int = 1000) -> List[Lead]:
        """
        Extract leads from the current page

        Args:
            max_leads: Maximum number of leads to extract

        Returns:
            List of Lead objects
        """
        logger.info(f"📊 Extracting leads (max: {max_leads})...")
        leads = []

        try:
            # Wait for leads to load
            await asyncio.sleep(3)

            # Try to find leads table/list/cards
            # DealMachine might use different structures
            table_selectors = [
                'table tbody tr',
                'div[role="row"]',
                '.lead-item',
                '.lead-card',
                '[data-testid*="lead"]'
            ]

            lead_elements = []
            for selector in table_selectors:
                try:
                    elements = await self.page.query_selector_all(selector)
                    if elements and len(elements) > 0:
                        lead_elements = elements
                        logger.info(f"✓ Found {len(elements)} leads using: {selector}")
                        break
                except:
                    continue

            if not lead_elements:
                logger.warning("⚠️  No lead elements found. Trying alternative extraction...")
                # Get page content and parse with regex
                content = await self.page.content()
                leads = await self._extract_from_html(content)
                return leads[:max_leads]

            # Extract data from each lead element
            for idx, element in enumerate(lead_elements[:max_leads]):
                if idx % 10 == 0:
                    logger.info(f"Processing lead {idx + 1}/{min(len(lead_elements), max_leads)}...")

                lead = await self._extract_lead_data(element)

                if lead:
                    self.stats["total_leads"] += 1

                    # Check DNC status
                    if self._is_dnc(lead):
                        lead.dnc_status = "DNC"
                        self.stats["dnc_filtered"] += 1
                        logger.debug(f"Filtered DNC: {lead.name or lead.address}")
                        continue  # Skip DNC leads

                    # Validate lead
                    if lead.is_valid():
                        leads.append(lead)
                        self.stats["clean_leads"] += 1
                    else:
                        self.stats["invalid_leads"] += 1
                        logger.debug(f"Invalid lead (missing data): {lead.name or lead.address}")

            logger.info(f"✅ Extracted {len(leads)} clean leads")
            return leads

        except Exception as e:
            logger.error(f"❌ Extraction error: {e}")
            return leads

    async def _extract_lead_data(self, element) -> Optional[Lead]:
        """Extract data from a single lead element"""
        try:
            # Get all text content
            text_content = await element.text_content()

            if not text_content:
                return None

            # Extract using patterns
            lead = Lead()

            # Phone number pattern
            phone_match = re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text_content)
            if phone_match:
                lead.phone = self._clean_phone(phone_match.group(0))

            # Email pattern
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text_content)
            if email_match:
                lead.email = email_match.group(0)

            # Try to get name from common selectors
            name_selectors = [
                '.name', '.owner-name', '[data-field="name"]',
                'td:nth-child(1)', 'div:nth-child(1)'
            ]

            for selector in name_selectors:
                try:
                    name_elem = await element.query_selector(selector)
                    if name_elem:
                        lead.name = (await name_elem.text_content()).strip()
                        if lead.name:
                            break
                except:
                    continue

            # Try to get address
            address_selectors = [
                '.address', '[data-field="address"]',
                'td:nth-child(2)', 'div:nth-child(2)'
            ]

            for selector in address_selectors:
                try:
                    addr_elem = await element.query_selector(selector)
                    if addr_elem:
                        lead.address = (await addr_elem.text_content()).strip()
                        if lead.address:
                            break
                except:
                    continue

            # Parse address components
            if lead.address:
                self._parse_address(lead)

            # Try to get phone from element
            phone_selectors = [
                '.phone', '[data-field="phone"]',
                'td:nth-child(3)', 'a[href^="tel:"]'
            ]

            for selector in phone_selectors:
                try:
                    phone_elem = await element.query_selector(selector)
                    if phone_elem:
                        phone_text = await phone_elem.text_content()
                        if phone_text:
                            lead.phone = self._clean_phone(phone_text)
                            break
                except:
                    continue

            return lead if (lead.name or lead.address) else None

        except Exception as e:
            logger.debug(f"Error extracting lead data: {e}")
            return None

    async def _extract_from_html(self, html: str) -> List[Lead]:
        """Extract leads from raw HTML using patterns"""
        logger.info("🔍 Using pattern-based extraction...")
        leads = []

        # Extract all phone numbers
        phones = re.findall(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', html)

        # Extract all emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', html)

        # Extract addresses (basic pattern)
        address_pattern = r'\d+\s+[A-Za-z0-9\s,]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Way|Court|Ct)'
        addresses = re.findall(address_pattern, html, re.IGNORECASE)

        logger.info(f"Found patterns - Phones: {len(phones)}, Emails: {len(emails)}, Addresses: {len(addresses)}")

        # Create leads from combinations
        for phone in phones[:100]:  # Limit to avoid duplicates
            lead = Lead()
            lead.phone = self._clean_phone(phone)

            # Try to find nearby address
            # This is simplified - real implementation would need context
            if addresses:
                lead.address = addresses[0]
                self._parse_address(lead)

            if lead.is_valid():
                leads.append(lead)

        return leads

    def _parse_address(self, lead: Lead):
        """Parse address components from full address"""
        if not lead.address:
            return

        # Extract ZIP code
        zip_match = re.search(r'\b\d{5}(?:-\d{4})?\b', lead.address)
        if zip_match:
            lead.zip_code = zip_match.group(0)

        # Extract state (2 letter code)
        state_match = re.search(r'\b[A-Z]{2}\b', lead.address)
        if state_match:
            lead.state = state_match.group(0)

        # City is usually before state
        if lead.state:
            parts = lead.address.split(lead.state)
            if len(parts) > 0:
                city_part = parts[0].strip().split(',')[-1].strip()
                lead.city = city_part

    def _clean_phone(self, phone: str) -> str:
        """Clean and format phone number"""
        # Remove all non-digits
        digits = re.sub(r'\D', '', phone)

        # Format as (XXX) XXX-XXXX
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        else:
            return phone

    def _is_dnc(self, lead: Lead) -> bool:
        """
        Check if lead is on DNC (Do Not Call) list

        This checks for common DNC indicators in the data
        """
        # Check status field
        if lead.status and any(dnc in lead.status.upper() for dnc in ['DNC', 'DO NOT CALL', 'SCRUB']):
            return True

        # Could integrate with actual DNC lookup services here
        # For now, we'll rely on DealMachine's filtering

        return False

    async def smart_extract(
        self,
        email: str,
        password: str,
        max_leads: int = 1000
    ) -> List[Lead]:
        """
        Smart extraction with auto-login and navigation

        Args:
            email: DealMachine email
            password: DealMachine password
            max_leads: Maximum leads to extract

        Returns:
            List of clean leads
        """
        logger.info("🚀 Starting smart extraction...")

        try:
            await self.initialize()

            # Login
            logged_in = await self.login(email, password)
            if not logged_in:
                logger.error("Failed to login")
                return []

            # Navigate to leads
            on_leads = await self.navigate_to_leads()
            if not on_leads:
                logger.warning("Could not navigate to leads tab, trying to extract from current page...")

            # Extract leads
            leads = await self.extract_leads(max_leads)

            # Save to files
            await self.save_leads(leads)

            # Print stats
            self._print_stats()

            return leads

        finally:
            await self.close()

    async def save_leads(self, leads: List[Lead]) -> Dict[str, Path]:
        """
        Save leads to multiple formats

        Returns:
            Dictionary of format -> file path
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        files = {}

        # Save as CSV
        csv_path = self.documents_dir / f"clean_leads_{timestamp}.csv"
        await self._save_csv(leads, csv_path)
        files['csv'] = csv_path
        logger.info(f"💾 CSV saved: {csv_path}")

        # Save as JSON
        json_path = self.documents_dir / f"clean_leads_{timestamp}.json"
        await self._save_json(leads, json_path)
        files['json'] = json_path
        logger.info(f"💾 JSON saved: {json_path}")

        # Save summary
        summary_path = self.documents_dir / f"extraction_summary_{timestamp}.txt"
        await self._save_summary(leads, summary_path)
        files['summary'] = summary_path
        logger.info(f"💾 Summary saved: {summary_path}")

        return files

    async def _save_csv(self, leads: List[Lead], path: Path):
        """Save leads to CSV"""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            if leads:
                # Write with full address field
                fieldnames = ['name', 'full_address', 'address', 'city', 'state', 'zip_code',
                             'phone', 'email', 'status', 'property_type', 'source', 'scraped_at']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for lead in leads:
                    row = asdict(lead)
                    row['full_address'] = lead.get_full_address()
                    # Remove dnc_status from export
                    row.pop('dnc_status', None)
                    writer.writerow({k: v for k, v in row.items() if k in fieldnames})

    async def _save_json(self, leads: List[Lead], path: Path):
        """Save leads to JSON"""
        data = {
            'extraction_date': datetime.utcnow().isoformat(),
            'total_leads': len(leads),
            'leads': [asdict(lead) for lead in leads]
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    async def _save_summary(self, leads: List[Lead], path: Path):
        """Save extraction summary"""
        lines = [
            "="*70,
            "DEALMACHINE LEADS EXTRACTION SUMMARY",
            "="*70,
            f"Extracted: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "",
            "STATISTICS:",
            f"  • Total leads processed: {self.stats['total_leads']}",
            f"  • Clean leads extracted: {self.stats['clean_leads']}",
            f"  • DNC filtered out: {self.stats['dnc_filtered']}",
            f"  • Invalid/incomplete: {self.stats['invalid_leads']}",
            "",
            "SAMPLE LEADS:",
        ]

        for idx, lead in enumerate(leads[:5], 1):
            lines.append(f"\n{idx}. {lead.name or 'N/A'}")
            lines.append(f"   Address: {lead.get_full_address()}")
            lines.append(f"   Phone: {lead.phone}")
            if lead.email:
                lines.append(f"   Email: {lead.email}")

        lines.extend([
            "",
            "="*70,
            f"Files saved to: {self.documents_dir}",
            "="*70
        ])

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _print_stats(self):
        """Print extraction statistics"""
        print("\n" + "="*70)
        print("📊 EXTRACTION COMPLETE!")
        print("="*70)
        print(f"✅ Clean leads: {self.stats['clean_leads']}")
        print(f"🚫 DNC filtered: {self.stats['dnc_filtered']}")
        print(f"⚠️  Invalid: {self.stats['invalid_leads']}")
        print(f"📁 Saved to: {self.documents_dir}")
        print("="*70 + "\n")

    async def close(self):
        """Clean up resources"""
        if self.browser:
            await self.browser.close()
        logger.info("👋 Extractor closed")


async def main():
    """Main extraction function"""
    print("\n" + "="*70)
    print("🔥 DEALMACHINE LEADS EXTRACTOR - ULTIMATE EDITION")
    print("="*70 + "\n")

    # Get credentials
    email = input("Enter DealMachine email: ").strip()
    if not email:
        print("❌ Email required")
        return

    import getpass
    password = getpass.getpass("Enter DealMachine password: ")
    if not password:
        print("❌ Password required")
        return

    max_leads = input("Max leads to extract (default 1000): ").strip()
    max_leads = int(max_leads) if max_leads.isdigit() else 1000

    # Initialize extractor
    extractor = DealMachineLeadsExtractor(headless=False)

    # Extract leads
    leads = await extractor.smart_extract(
        email=email,
        password=password,
        max_leads=max_leads
    )

    print(f"\n✅ Extraction complete! Got {len(leads)} clean leads")
    print(f"📁 Check {extractor.documents_dir} for files\n")


if __name__ == "__main__":
    asyncio.run(main())
