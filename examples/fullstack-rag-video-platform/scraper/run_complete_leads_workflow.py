#!/usr/bin/env python3
"""
COMPLETE LEADS WORKFLOW - ALL-IN-ONE
Extract -> Clean -> Deduplicate -> Organize -> Export

This is the ULTIMATE leads management system that:
1. Extracts clean leads from DealMachine (name, address, phone)
2. Filters out DNC scrubs automatically
3. Deduplicates across multiple sources
4. Cleans and standardizes data
5. Organizes by city/state
6. Exports to beautifully organized CSV files
"""

import asyncio
import sys
from pathlib import Path

# Add scraper directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dealmachine_leads_extractor import DealMachineLeadsExtractor
from csv_leads_manager import CSVLeadsManager


async def run_complete_workflow(email: str, password: str, max_leads: int = 1000):
    """
    Run the complete leads workflow

    Args:
        email: DealMachine email
        password: DealMachine password
        max_leads: Max leads to extract
    """
    print("\n" + "="*70)
    print("🔥 COMPLETE LEADS WORKFLOW - ULTIMATE EDITION")
    print("="*70 + "\n")

    print("This workflow will:")
    print("  ✅ Extract clean leads from DealMachine")
    print("  ✅ Filter out DNC scrubs")
    print("  ✅ Deduplicate leads")
    print("  ✅ Clean and standardize data")
    print("  ✅ Organize by city and state")
    print("  ✅ Export to CSV files")
    print("\n" + "="*70 + "\n")

    # STEP 1: Extract leads from DealMachine
    print("STEP 1: Extracting leads from DealMachine...")
    print("-" * 70)

    extractor = DealMachineLeadsExtractor(
        documents_dir="/home/user/Documents/dealmachine_data/leads/raw"
    )

    leads = await extractor.smart_extract(
        email=email,
        password=password,
        max_leads=max_leads
    )

    if not leads:
        print("\n❌ No leads extracted. Please check your credentials and try again.\n")
        return

    print(f"\n✅ STEP 1 COMPLETE: Extracted {len(leads)} clean leads\n")
    await asyncio.sleep(2)

    # STEP 2: Import into CSV Manager
    print("\nSTEP 2: Importing into CSV Manager...")
    print("-" * 70)

    manager = CSVLeadsManager(
        base_dir="/home/user/Documents/dealmachine_data/leads/organized"
    )

    # Find the latest CSV file
    raw_dir = Path("/home/user/Documents/dealmachine_data/leads/raw")
    csv_files = list(raw_dir.glob("clean_leads_*.csv"))

    if csv_files:
        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
        manager.import_csv(str(latest_csv))
        print(f"✅ Imported from: {latest_csv.name}")
    else:
        print("⚠️  No CSV files found, using extracted leads")

    print(f"\n✅ STEP 2 COMPLETE: {len(manager.leads)} leads imported\n")
    await asyncio.sleep(1)

    # STEP 3: Deduplicate
    print("\nSTEP 3: Deduplicating leads...")
    print("-" * 70)

    duplicates = manager.deduplicate()
    print(f"✅ Removed {duplicates} duplicate leads")
    print(f"✅ Remaining: {len(manager.leads)} unique leads")

    print(f"\n✅ STEP 3 COMPLETE\n")
    await asyncio.sleep(1)

    # STEP 4: Clean data
    print("\nSTEP 4: Cleaning and standardizing data...")
    print("-" * 70)

    manager.clean_data()
    print("✅ Data cleaned and standardized")

    print(f"\n✅ STEP 4 COMPLETE\n")
    await asyncio.sleep(1)

    # STEP 5: Organize and Export
    print("\nSTEP 5: Organizing and exporting...")
    print("-" * 70)

    # Export master CSV
    master_file = manager.export_master_csv()
    print(f"✅ Master CSV: {master_file.name}")

    # Export by state
    state_files = manager.export_by_state()
    print(f"✅ Exported {len(state_files)} state files")

    # Export by city
    city_files = manager.export_by_city()
    print(f"✅ Exported {len(city_files)} city files")

    # Generate report
    report_file = manager.save_report()
    print(f"✅ Report: {report_file.name}")

    print(f"\n✅ STEP 5 COMPLETE\n")
    await asyncio.sleep(1)

    # FINAL SUMMARY
    print("\n" + "="*70)
    print("🎉 WORKFLOW COMPLETE!")
    print("="*70)

    print(manager.generate_report())

    print("\n📁 ALL FILES SAVED TO:")
    print(f"   Raw data: /home/user/Documents/dealmachine_data/leads/raw/")
    print(f"   Organized: /home/user/Documents/dealmachine_data/leads/organized/")

    print("\n📊 EXPORTED FILES:")
    print(f"   • 1 Master CSV with all {len(manager.leads)} leads")
    print(f"   • {len(state_files)} State-specific CSVs")
    print(f"   • {len(city_files)} City-specific CSVs")
    print(f"   • 1 Comprehensive report")

    print("\n✨ Your leads are ready to use!")
    print("="*70 + "\n")


async def run_demo_workflow():
    """Run a demo workflow with simulated data"""
    print("\n" + "="*70)
    print("🎬 DEMO MODE - Sample Workflow")
    print("="*70 + "\n")

    print("Creating sample leads for demonstration...")

    # Create sample CSV
    from csv_leads_manager import Lead, CSVLeadsManager
    import csv

    sample_leads = [
        Lead(
            name="John Smith",
            address="123 Main St",
            city="Austin",
            state="TX",
            zip_code="78701",
            phone="(512) 555-0101",
            email="john@example.com",
            source="Demo"
        ),
        Lead(
            name="Jane Doe",
            address="456 Oak Ave",
            city="Austin",
            state="TX",
            zip_code="78702",
            phone="(512) 555-0102",
            source="Demo"
        ),
        Lead(
            name="Bob Johnson",
            address="789 Pine St",
            city="Dallas",
            state="TX",
            zip_code="75201",
            phone="(214) 555-0103",
            email="bob@example.com",
            source="Demo"
        ),
        Lead(
            name="Alice Williams",
            address="321 Elm St",
            city="Houston",
            state="TX",
            zip_code="77001",
            phone="(713) 555-0104",
            source="Demo"
        ),
        Lead(
            name="Charlie Brown",
            address="654 Maple Dr",
            city="Houston",
            state="TX",
            zip_code="77002",
            phone="(713) 555-0105",
            email="charlie@example.com",
            source="Demo"
        ),
    ]

    # Create sample CSV file
    demo_dir = Path("/home/user/Documents/dealmachine_data/leads/demo")
    demo_dir.mkdir(parents=True, exist_ok=True)

    sample_csv = demo_dir / "sample_leads.csv"
    with open(sample_csv, 'w', newline='') as f:
        fieldnames = ['name', 'address', 'city', 'state', 'zip_code', 'phone', 'email', 'source']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for lead in sample_leads:
            writer.writerow({k: v for k, v in lead.to_dict().items() if k in fieldnames})

    print(f"✅ Created sample CSV: {sample_csv}")

    # Run manager workflow
    manager = CSVLeadsManager(
        base_dir="/home/user/Documents/dealmachine_data/leads/demo_organized"
    )

    print("\n📥 Importing sample leads...")
    manager.import_csv(str(sample_csv))

    print("\n🔍 Deduplicating...")
    manager.deduplicate()

    print("\n🧹 Cleaning data...")
    manager.clean_data()

    print("\n📤 Exporting...")
    master_file = manager.export_master_csv()
    state_files = manager.export_by_state()
    city_files = manager.export_by_city()
    report_file = manager.save_report()

    print("\n" + "="*70)
    print(manager.generate_report())

    print("\n✅ Demo complete! Check the demo_organized folder for results.")
    print("="*70 + "\n")


async def main():
    """Main entry point"""
    import sys

    print("\n" + "="*70)
    print("🚀 DEALMACHINE LEADS - COMPLETE WORKFLOW")
    print("="*70 + "\n")

    print("Choose mode:")
    print("1. Full workflow (requires DealMachine credentials)")
    print("2. Demo mode (sample data)")
    print("3. CSV Manager only (manage existing CSVs)")

    choice = input("\nChoice (1/2/3): ").strip()

    if choice == "1":
        # Full workflow
        email = input("\nEnter DealMachine email: ").strip()
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

        await run_complete_workflow(email, password, max_leads)

    elif choice == "2":
        # Demo mode
        await run_demo_workflow()

    elif choice == "3":
        # CSV Manager only
        from csv_leads_manager import main as csv_main
        csv_main()

    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())
