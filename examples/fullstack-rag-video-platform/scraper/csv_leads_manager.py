#!/usr/bin/env python3
"""
CSV Leads Manager - ULTIMATE ORGANIZATION TOOL
Import, export, merge, deduplicate, and organize your leads like a pro
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("csv-manager")


@dataclass
class Lead:
    """Lead data structure"""
    name: str = ""
    full_address: str = ""
    address: str = ""
    city: str = ""
    state: str = ""
    zip_code: str = ""
    phone: str = ""
    email: str = ""
    status: str = ""
    property_type: str = ""
    source: str = ""
    scraped_at: str = ""
    notes: str = ""
    tags: str = ""

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary"""
        return {k: v for k, v in asdict(self).items() if v}

    def get_unique_key(self) -> str:
        """Get unique identifier for deduplication"""
        # Clean phone for comparison
        phone_clean = re.sub(r'\D', '', self.phone)
        # Use phone + address as unique key
        return f"{phone_clean}_{self.address.lower().strip()}"


class CSVLeadsManager:
    """
    Ultimate CSV leads management system
    Features:
    - Import from multiple CSV files
    - Export to organized CSV files
    - Merge multiple lead lists
    - Deduplicate leads
    - Clean and standardize data
    - Organize by city, state, status
    - Filter and search
    - Tag management
    """

    def __init__(self, base_dir: str = "/home/user/Documents/dealmachine_data/organized"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Storage
        self.leads: List[Lead] = []
        self.original_count = 0

        # Statistics
        self.stats = {
            "imported": 0,
            "duplicates_removed": 0,
            "cleaned": 0,
            "exported": 0
        }

        logger.info("📊 CSV Leads Manager initialized")
        logger.info(f"📁 Base directory: {self.base_dir}")

    def import_csv(self, file_path: str) -> int:
        """
        Import leads from CSV file

        Args:
            file_path: Path to CSV file

        Returns:
            Number of leads imported
        """
        logger.info(f"📥 Importing from: {file_path}")

        path = Path(file_path)
        if not path.exists():
            logger.error(f"❌ File not found: {file_path}")
            return 0

        try:
            count = 0
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    lead = self._row_to_lead(row)
                    if lead:
                        self.leads.append(lead)
                        count += 1

            self.stats["imported"] += count
            logger.info(f"✅ Imported {count} leads")
            return count

        except Exception as e:
            logger.error(f"❌ Import error: {e}")
            return 0

    def import_multiple(self, file_patterns: List[str]) -> int:
        """
        Import from multiple CSV files using glob patterns

        Args:
            file_patterns: List of file paths or glob patterns

        Returns:
            Total leads imported
        """
        logger.info("📥 Importing from multiple files...")

        total = 0
        for pattern in file_patterns:
            # Expand glob pattern
            for file_path in Path().glob(pattern):
                total += self.import_csv(str(file_path))

        logger.info(f"✅ Total imported: {total} leads from {len(file_patterns)} patterns")
        return total

    def _row_to_lead(self, row: Dict[str, str]) -> Optional[Lead]:
        """Convert CSV row to Lead object"""
        try:
            # Map various CSV column names to Lead fields
            lead = Lead()

            # Name mapping
            for name_field in ['name', 'owner_name', 'owner', 'contact_name', 'full_name']:
                if name_field in row:
                    lead.name = row[name_field].strip()
                    break

            # Address mappings
            for addr_field in ['full_address', 'address', 'property_address', 'street_address']:
                if addr_field in row:
                    lead.full_address = row[addr_field].strip()
                    if not lead.address:
                        lead.address = row[addr_field].strip()
                    break

            if 'address' in row:
                lead.address = row['address'].strip()

            lead.city = row.get('city', '').strip()
            lead.state = row.get('state', '').strip()
            lead.zip_code = row.get('zip_code', row.get('zip', '')).strip()

            # Phone mapping
            for phone_field in ['phone', 'phone_number', 'mobile', 'cell', 'contact_phone']:
                if phone_field in row and row[phone_field]:
                    lead.phone = self._clean_phone(row[phone_field])
                    break

            # Email mapping
            for email_field in ['email', 'email_address', 'contact_email']:
                if email_field in row:
                    lead.email = row[email_field].strip()
                    break

            lead.status = row.get('status', '').strip()
            lead.property_type = row.get('property_type', '').strip()
            lead.source = row.get('source', 'CSV Import').strip()
            lead.notes = row.get('notes', '').strip()
            lead.tags = row.get('tags', '').strip()

            # Set scraped_at
            lead.scraped_at = row.get('scraped_at', datetime.utcnow().isoformat())

            # Only return if we have minimum data
            if lead.name or lead.address or lead.phone:
                return lead

            return None

        except Exception as e:
            logger.debug(f"Error converting row: {e}")
            return None

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

    def deduplicate(self, keep: str = 'first') -> int:
        """
        Remove duplicate leads

        Args:
            keep: Which duplicate to keep ('first' or 'last')

        Returns:
            Number of duplicates removed
        """
        logger.info("🔍 Deduplicating leads...")

        original_count = len(self.leads)
        seen: Set[str] = set()
        unique_leads = []

        for lead in self.leads:
            key = lead.get_unique_key()

            if key not in seen:
                seen.add(key)
                unique_leads.append(lead)
            elif keep == 'last':
                # Remove old, add new
                unique_leads = [l for l in unique_leads if l.get_unique_key() != key]
                unique_leads.append(lead)

        duplicates = original_count - len(unique_leads)
        self.leads = unique_leads
        self.stats["duplicates_removed"] += duplicates

        logger.info(f"✅ Removed {duplicates} duplicates")
        return duplicates

    def clean_data(self):
        """Clean and standardize lead data"""
        logger.info("🧹 Cleaning data...")

        cleaned = 0
        for lead in self.leads:
            # Standardize phone
            if lead.phone:
                old_phone = lead.phone
                lead.phone = self._clean_phone(lead.phone)
                if old_phone != lead.phone:
                    cleaned += 1

            # Standardize state to uppercase
            if lead.state:
                lead.state = lead.state.upper()

            # Clean whitespace
            lead.name = lead.name.strip()
            lead.address = lead.address.strip()
            lead.city = lead.city.strip()

            # Build full address if missing
            if not lead.full_address and (lead.address or lead.city):
                parts = [lead.address, lead.city, lead.state, lead.zip_code]
                lead.full_address = ", ".join([p for p in parts if p])

        self.stats["cleaned"] = cleaned
        logger.info(f"✅ Cleaned {cleaned} records")

    def filter_leads(
        self,
        city: Optional[str] = None,
        state: Optional[str] = None,
        status: Optional[str] = None,
        has_phone: bool = True,
        has_email: bool = False
    ) -> List[Lead]:
        """
        Filter leads by criteria

        Args:
            city: Filter by city
            state: Filter by state
            status: Filter by status
            has_phone: Only include leads with phone
            has_email: Only include leads with email

        Returns:
            Filtered list of leads
        """
        filtered = self.leads

        if city:
            filtered = [l for l in filtered if l.city.lower() == city.lower()]

        if state:
            filtered = [l for l in filtered if l.state.upper() == state.upper()]

        if status:
            filtered = [l for l in filtered if l.status.lower() == status.lower()]

        if has_phone:
            filtered = [l for l in filtered if l.phone]

        if has_email:
            filtered = [l for l in filtered if l.email]

        logger.info(f"🔍 Filtered: {len(filtered)} leads match criteria")
        return filtered

    def organize_by_city(self) -> Dict[str, List[Lead]]:
        """
        Organize leads by city

        Returns:
            Dictionary mapping city -> leads
        """
        logger.info("📂 Organizing by city...")

        by_city: Dict[str, List[Lead]] = defaultdict(list)

        for lead in self.leads:
            city = lead.city if lead.city else "Unknown"
            by_city[city].append(lead)

        logger.info(f"✅ Organized into {len(by_city)} cities")
        return dict(by_city)

    def organize_by_state(self) -> Dict[str, List[Lead]]:
        """
        Organize leads by state

        Returns:
            Dictionary mapping state -> leads
        """
        logger.info("📂 Organizing by state...")

        by_state: Dict[str, List[Lead]] = defaultdict(list)

        for lead in self.leads:
            state = lead.state if lead.state else "Unknown"
            by_state[state].append(lead)

        logger.info(f"✅ Organized into {len(by_state)} states")
        return dict(by_state)

    def export_csv(
        self,
        filename: str,
        leads: Optional[List[Lead]] = None
    ) -> Path:
        """
        Export leads to CSV

        Args:
            filename: Output filename
            leads: Leads to export (default: all)

        Returns:
            Path to exported file
        """
        if leads is None:
            leads = self.leads

        output_path = self.base_dir / filename

        logger.info(f"📤 Exporting {len(leads)} leads to: {output_path}")

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if leads:
                # Collect all unique fieldnames from all leads
                all_fieldnames = set()
                for lead in leads:
                    all_fieldnames.update(lead.to_dict().keys())

                # Sort fieldnames for consistent column order
                fieldnames = sorted(all_fieldnames)

                # Move important fields to front
                priority_fields = ['name', 'full_address', 'address', 'city', 'state', 'zip_code', 'phone', 'email']
                ordered_fieldnames = []
                for field in priority_fields:
                    if field in fieldnames:
                        ordered_fieldnames.append(field)
                        fieldnames.remove(field)
                ordered_fieldnames.extend(sorted(fieldnames))

                writer = csv.DictWriter(f, fieldnames=ordered_fieldnames, extrasaction='ignore')
                writer.writeheader()

                for lead in leads:
                    writer.writerow(lead.to_dict())

        self.stats["exported"] += len(leads)
        logger.info(f"✅ Exported {len(leads)} leads")
        return output_path

    def export_by_city(self) -> List[Path]:
        """
        Export separate CSV files for each city

        Returns:
            List of exported file paths
        """
        logger.info("📤 Exporting by city...")

        by_city = self.organize_by_city()
        files = []

        for city, city_leads in by_city.items():
            # Clean city name for filename
            safe_city = re.sub(r'[^a-zA-Z0-9_-]', '_', city)
            filename = f"leads_{safe_city}_{len(city_leads)}.csv"
            path = self.export_csv(filename, city_leads)
            files.append(path)

        logger.info(f"✅ Exported {len(files)} city files")
        return files

    def export_by_state(self) -> List[Path]:
        """
        Export separate CSV files for each state

        Returns:
            List of exported file paths
        """
        logger.info("📤 Exporting by state...")

        by_state = self.organize_by_state()
        files = []

        for state, state_leads in by_state.items():
            filename = f"leads_{state}_{len(state_leads)}.csv"
            path = self.export_csv(filename, state_leads)
            files.append(path)

        logger.info(f"✅ Exported {len(files)} state files")
        return files

    def export_master_csv(self) -> Path:
        """
        Export master CSV with all leads

        Returns:
            Path to master CSV
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"MASTER_LEADS_{timestamp}_{len(self.leads)}.csv"
        return self.export_csv(filename)

    def generate_report(self) -> str:
        """
        Generate comprehensive report

        Returns:
            Report text
        """
        by_city = self.organize_by_city()
        by_state = self.organize_by_state()

        # Count leads with phone/email
        with_phone = len([l for l in self.leads if l.phone])
        with_email = len([l for l in self.leads if l.email])
        with_both = len([l for l in self.leads if l.phone and l.email])

        lines = [
            "="*70,
            "LEADS DATABASE REPORT",
            "="*70,
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC",
            "",
            "STATISTICS:",
            f"  • Total leads: {len(self.leads)}",
            f"  • Leads with phone: {with_phone} ({with_phone/len(self.leads)*100:.1f}%)" if self.leads else "  • Leads with phone: 0",
            f"  • Leads with email: {with_email} ({with_email/len(self.leads)*100:.1f}%)" if self.leads else "  • Leads with email: 0",
            f"  • Leads with both: {with_both}",
            "",
            "IMPORT/EXPORT:",
            f"  • Total imported: {self.stats['imported']}",
            f"  • Duplicates removed: {self.stats['duplicates_removed']}",
            f"  • Records cleaned: {self.stats['cleaned']}",
            f"  • Total exported: {self.stats['exported']}",
            "",
            f"DISTRIBUTION BY STATE ({len(by_state)} states):",
        ]

        # Top states
        sorted_states = sorted(by_state.items(), key=lambda x: len(x[1]), reverse=True)
        for state, state_leads in sorted_states[:10]:
            lines.append(f"  • {state}: {len(state_leads)} leads")

        if len(sorted_states) > 10:
            lines.append(f"  • ... and {len(sorted_states) - 10} more states")

        lines.extend([
            "",
            f"DISTRIBUTION BY CITY ({len(by_city)} cities):",
        ])

        # Top cities
        sorted_cities = sorted(by_city.items(), key=lambda x: len(x[1]), reverse=True)
        for city, city_leads in sorted_cities[:15]:
            lines.append(f"  • {city}: {len(city_leads)} leads")

        if len(sorted_cities) > 15:
            lines.append(f"  • ... and {len(sorted_cities) - 15} more cities")

        lines.extend([
            "",
            "="*70,
            f"Files location: {self.base_dir}",
            "="*70
        ])

        return "\n".join(lines)

    def save_report(self, filename: str = "leads_report.txt") -> Path:
        """Save report to file"""
        report = self.generate_report()
        path = self.base_dir / filename

        with open(path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"📊 Report saved: {path}")
        return path

    def get_summary(self):
        """Print summary to console"""
        print("\n" + "="*70)
        print("📊 LEADS MANAGER SUMMARY")
        print("="*70)
        print(f"Total leads: {len(self.leads)}")
        print(f"Imported: {self.stats['imported']}")
        print(f"Duplicates removed: {self.stats['duplicates_removed']}")
        print(f"Exported: {self.stats['exported']}")
        print("="*70 + "\n")


def main():
    """Interactive CLI for CSV management"""
    print("\n" + "="*70)
    print("🚀 CSV LEADS MANAGER - ULTIMATE EDITION")
    print("="*70 + "\n")

    manager = CSVLeadsManager()

    while True:
        print("\nWhat would you like to do?")
        print("1. Import CSV file(s)")
        print("2. View statistics")
        print("3. Deduplicate leads")
        print("4. Clean data")
        print("5. Export master CSV")
        print("6. Export by city")
        print("7. Export by state")
        print("8. Generate report")
        print("9. Filter and export")
        print("0. Exit")

        choice = input("\nChoice: ").strip()

        if choice == "1":
            path = input("Enter CSV file path or pattern: ").strip()
            manager.import_csv(path)

        elif choice == "2":
            manager.get_summary()

        elif choice == "3":
            manager.deduplicate()
            manager.get_summary()

        elif choice == "4":
            manager.clean_data()

        elif choice == "5":
            path = manager.export_master_csv()
            print(f"✅ Exported to: {path}")

        elif choice == "6":
            files = manager.export_by_city()
            print(f"✅ Exported {len(files)} city files")

        elif choice == "7":
            files = manager.export_by_state()
            print(f"✅ Exported {len(files)} state files")

        elif choice == "8":
            print(manager.generate_report())
            manager.save_report()

        elif choice == "9":
            city = input("Filter by city (or press Enter): ").strip() or None
            state = input("Filter by state (or press Enter): ").strip() or None
            filtered = manager.filter_leads(city=city, state=state)
            filename = input("Export filename: ").strip()
            manager.export_csv(filename, filtered)

        elif choice == "0":
            print("\n👋 Goodbye!\n")
            break

        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()
