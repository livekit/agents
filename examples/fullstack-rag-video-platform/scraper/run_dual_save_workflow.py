#!/usr/bin/env python3
"""
DUAL-SAVE VERSION - Saves to both WSL and Windows locations
Workaround for Windows 11 KB5065426 network share issues
"""

import asyncio
import sys
import shutil
from pathlib import Path

# Add scraper directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dealmachine_leads_extractor import DealMachineLeadsExtractor
from csv_leads_manager import CSVLeadsManager


def get_save_locations():
    """
    Get both WSL and Windows save locations
    """
    # WSL location (in your home directory)
    wsl_base = Path.home() / "Documents/dealmachine_data/leads"

    # Windows location (accessible via /mnt/c/)
    windows_base = Path("/mnt/c/Users") / Path.home().name / "Documents/dealmachine_data/leads"

    # Fallback: Try to find Windows user directory
    if not windows_base.exists():
        # Try common Windows paths
        possible_paths = [
            Path("/mnt/c/Users/user/Documents/dealmachine_data/leads"),
            Path("/mnt/c/Users/Public/Documents/dealmachine_data/leads"),
        ]

        for p in possible_paths:
            if p.parent.parent.exists():  # If Documents folder exists
                windows_base = p
                break

    return {
        'wsl': wsl_base,
        'windows': windows_base
    }


def copy_to_windows(wsl_path: Path, windows_path: Path):
    """
    Copy files from WSL to Windows location
    """
    try:
        # Create Windows directory
        windows_path.mkdir(parents=True, exist_ok=True)

        # Copy all files from WSL to Windows
        if wsl_path.exists():
            for item in wsl_path.rglob('*'):
                if item.is_file():
                    # Calculate relative path
                    rel_path = item.relative_to(wsl_path)
                    dest = windows_path / rel_path

                    # Create parent directory
                    dest.parent.mkdir(parents=True, exist_ok=True)

                    # Copy file
                    shutil.copy2(item, dest)
                    print(f"✅ Copied to Windows: {dest}")

        return True
    except Exception as e:
        print(f"⚠️  Warning: Could not copy to Windows location: {e}")
        return False


async def run_complete_workflow(email: str, password: str, max_leads: int = 1000):
    """
    Run complete workflow with dual save
    """
    locations = get_save_locations()

    print("\n" + "="*70)
    print("🔥 COMPLETE LEADS WORKFLOW - DUAL SAVE EDITION")
    print("="*70 + "\n")

    print("📁 Save Locations:")
    print(f"  WSL:     {locations['wsl']}")
    print(f"  Windows: {locations['windows']}")
    print()

    # Run extraction to WSL location
    extractor = DealMachineLeadsExtractor(
        documents_dir=str(locations['wsl'] / "raw")
    )

    leads = await extractor.smart_extract(
        email=email,
        password=password,
        max_leads=max_leads
    )

    if not leads:
        print("\n❌ No leads extracted.\n")
        return

    # Import and organize
    manager = CSVLeadsManager(
        base_dir=str(locations['wsl'] / "organized")
    )

    # Find latest CSV
    raw_dir = locations['wsl'] / "raw"
    csv_files = list(raw_dir.glob("clean_leads_*.csv"))

    if csv_files:
        latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
        manager.import_csv(str(latest_csv))

    # Deduplicate and clean
    manager.deduplicate()
    manager.clean_data()

    # Export
    master_file = manager.export_master_csv()
    state_files = manager.export_by_state()
    city_files = manager.export_by_city()
    report_file = manager.save_report()

    print("\n" + "="*70)
    print("🎉 EXTRACTION COMPLETE!")
    print("="*70)

    # Copy to Windows
    print("\n📋 Copying files to Windows location...")
    print("-" * 70)

    copy_success = copy_to_windows(
        locations['wsl'] / "organized",
        locations['windows'] / "organized"
    )

    copy_success = copy_to_windows(
        locations['wsl'] / "raw",
        locations['windows'] / "raw"
    ) and copy_success

    print("\n📁 FILES SAVED TO:")
    print(f"  ✅ WSL:     {locations['wsl']}")
    if copy_success:
        print(f"  ✅ Windows: {locations['windows']}")
    else:
        print(f"  ⚠️  Windows: Copy failed (check permissions)")

    print("\n💡 Access your files:")
    print(f"  • From WSL:     cd {locations['wsl']}/organized")
    print(f"  • From Windows: C:\\Users\\{Path.home().name}\\Documents\\dealmachine_data\\leads\\organized")
    print("="*70 + "\n")


async def run_demo_workflow():
    """
    Run demo with dual save
    """
    locations = get_save_locations()

    print("\n" + "="*70)
    print("🎬 DEMO MODE - DUAL SAVE")
    print("="*70 + "\n")

    print("Creating sample leads...")

    # Create demo directory
    demo_dir = locations['wsl'] / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Create sample CSV
    from csv_leads_manager import Lead
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
    ]

    sample_csv = demo_dir / "sample_leads.csv"
    with open(sample_csv, 'w', newline='') as f:
        fieldnames = ['name', 'address', 'city', 'state', 'zip_code', 'phone', 'email', 'source']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for lead in sample_leads:
            writer.writerow({k: v for k, v in lead.to_dict().items() if k in fieldnames})

    print(f"✅ Created sample CSV")

    # Process
    manager = CSVLeadsManager(
        base_dir=str(locations['wsl'] / "demo_organized")
    )

    manager.import_csv(str(sample_csv))
    manager.deduplicate()
    manager.clean_data()

    master_file = manager.export_master_csv()
    state_files = manager.export_by_state()
    city_files = manager.export_by_city()
    report_file = manager.save_report()

    print("\n📋 Copying to Windows...")

    copy_success = copy_to_windows(
        locations['wsl'] / "demo_organized",
        locations['windows'] / "demo_organized"
    )

    print("\n" + "="*70)
    print(manager.generate_report())

    print("\n📁 FILES SAVED TO:")
    print(f"  ✅ WSL:     {locations['wsl']}/demo_organized")
    if copy_success:
        print(f"  ✅ Windows: {locations['windows']}/demo_organized")

    print("\n💡 Access from Windows:")
    print(f"  C:\\Users\\{Path.home().name}\\Documents\\dealmachine_data\\leads\\demo_organized")
    print("="*70 + "\n")


async def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("🚀 DEALMACHINE LEADS - DUAL SAVE EDITION")
    print("   (Workaround for Windows 11 KB5065426 network share issue)")
    print("="*70 + "\n")

    print("Choose mode:")
    print("1. Full workflow (requires DealMachine credentials)")
    print("2. Demo mode (sample data)")
    print("3. CSV Manager only")

    choice = input("\nChoice (1/2/3): ").strip()

    if choice == "1":
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
        await run_demo_workflow()

    elif choice == "3":
        from csv_leads_manager import main as csv_main
        csv_main()

    else:
        print("Invalid choice")


if __name__ == "__main__":
    asyncio.run(main())
