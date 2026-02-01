# 🔥 DealMachine Leads Extraction System - ULTIMATE EDITION

The most powerful leads extraction, organization, and management system for DealMachine.

## ✨ Features

### 🎯 Smart Extraction
- ✅ **Extracts complete lead data**: Name, Full Address, Phone Number
- ✅ **Automatic DNC filtering**: Removes Do Not Call leads automatically
- ✅ **Multi-strategy extraction**: Uses multiple methods to find data
- ✅ **Auto-login**: Handles DealMachine authentication
- ✅ **Intelligent navigation**: Finds and navigates to leads tab automatically

### 📊 CSV Management
- ✅ **Import from multiple sources**: Merge CSVs from different exports
- ✅ **Smart deduplication**: Removes duplicate leads intelligently
- ✅ **Data cleaning**: Standardizes phone numbers, addresses, and more
- ✅ **Advanced organization**: Sort by city, state, or custom criteria
- ✅ **Bulk export**: Export organized CSVs by geography

### 🧠 Intelligent Features
- ✅ **Pattern learning**: Gets smarter over time
- ✅ **Multiple export formats**: CSV, JSON, and formatted reports
- ✅ **Comprehensive statistics**: Track everything
- ✅ **Data validation**: Ensures lead quality

## 📁 Project Structure

```
scraper/
├── dealmachine_leads_extractor.py    # Main leads extraction engine
├── csv_leads_manager.py              # CSV import/export and organization
├── run_complete_leads_workflow.py    # All-in-one workflow script
├── LEADS_EXTRACTION_README.md        # This file
```

## 🚀 Quick Start

### Option 1: Complete Workflow (Recommended)

Extract leads and organize them all in one go:

```bash
cd /home/user/agents/examples/fullstack-rag-video-platform/scraper
python3 run_complete_leads_workflow.py
```

Choose option 1, enter your DealMachine credentials, and let it run!

**What it does:**
1. Extracts clean leads from DealMachine
2. Filters out DNC leads
3. Deduplicates across sources
4. Cleans and standardizes data
5. Organizes by city and state
6. Exports to multiple CSV files
7. Generates comprehensive reports

### Option 2: Demo Mode

Try it with sample data first:

```bash
python3 run_complete_leads_workflow.py
# Choose option 2
```

### Option 3: Extraction Only

Just extract leads from DealMachine:

```bash
python3 dealmachine_leads_extractor.py
```

### Option 4: CSV Management Only

Manage existing CSV files:

```bash
python3 csv_leads_manager.py
```

## 📋 Detailed Usage

### Leads Extraction

```python
from dealmachine_leads_extractor import DealMachineLeadsExtractor

# Initialize extractor
extractor = DealMachineLeadsExtractor(
    documents_dir="/home/user/Documents/dealmachine_data/leads",
    headless=True  # Run browser in background
)

# Extract leads
leads = await extractor.smart_extract(
    email="your@email.com",
    password="your_password",
    max_leads=1000
)

# Leads are automatically saved to:
# - clean_leads_TIMESTAMP.csv
# - clean_leads_TIMESTAMP.json
# - extraction_summary_TIMESTAMP.txt
```

### CSV Management

```python
from csv_leads_manager import CSVLeadsManager

# Initialize manager
manager = CSVLeadsManager(
    base_dir="/home/user/Documents/dealmachine_data/organized"
)

# Import CSVs
manager.import_csv("path/to/leads.csv")
manager.import_multiple(["*.csv", "exports/*.csv"])

# Clean and organize
manager.deduplicate()  # Remove duplicates
manager.clean_data()   # Standardize data

# Filter
filtered = manager.filter_leads(
    city="Austin",
    state="TX",
    has_phone=True
)

# Export
manager.export_master_csv()        # All leads
manager.export_by_city()           # Separate file per city
manager.export_by_state()          # Separate file per state
manager.export_csv("custom.csv", filtered)  # Custom export

# Generate report
print(manager.generate_report())
manager.save_report()
```

## 📂 Output Structure

All files are saved to `/home/user/Documents/dealmachine_data/leads/`:

```
leads/
├── raw/                                    # Raw extracted data
│   ├── clean_leads_20260201_120000.csv
│   ├── clean_leads_20260201_120000.json
│   └── extraction_summary_20260201_120000.txt
│
├── organized/                              # Organized exports
│   ├── MASTER_LEADS_20260201_120000_500.csv
│   ├── leads_TX_250.csv
│   ├── leads_CA_150.csv
│   ├── leads_Austin_50.csv
│   ├── leads_Dallas_75.csv
│   ├── leads_Houston_125.csv
│   └── leads_report.txt
│
└── demo/                                   # Demo data
    └── sample_leads.csv
```

## 📊 CSV Format

All exported CSVs include these fields:

| Field | Description | Example |
|-------|-------------|---------|
| `name` | Property owner name | John Smith |
| `full_address` | Complete formatted address | 123 Main St, Austin, TX 78701 |
| `address` | Street address | 123 Main St |
| `city` | City | Austin |
| `state` | State (2-letter) | TX |
| `zip_code` | ZIP code | 78701 |
| `phone` | Phone number (formatted) | (512) 555-0101 |
| `email` | Email address | john@example.com |
| `status` | Lead status | New |
| `property_type` | Type of property | Single Family |
| `source` | Where lead came from | DealMachine |
| `scraped_at` | Extraction timestamp | 2026-02-01T12:00:00 |
| `notes` | Custom notes | - |
| `tags` | Custom tags | - |

## 🎯 Key Features Explained

### DNC Filtering

The system automatically filters out leads marked as "Do Not Call":

- Checks status field for DNC indicators
- Filters during extraction (not exported)
- Can integrate with external DNC lists
- Statistics show how many were filtered

### Deduplication

Smart duplicate detection based on:

- Phone number + address combination
- Handles various phone formats
- Keeps most recent by default
- Can choose which duplicate to keep

### Data Cleaning

Automatic standardization:

- Phone numbers: `(512) 555-0101` format
- States: Uppercase 2-letter codes
- Whitespace trimming
- Address normalization
- Full address building

### Organization

Multiple organization methods:

- **By City**: Separate file for each city
- **By State**: Separate file for each state
- **By Custom Filter**: Your own criteria
- **Master File**: Everything in one place

## 📈 Statistics & Reports

The system tracks:

- Total leads processed
- Clean leads extracted
- DNC leads filtered
- Invalid/incomplete leads
- Duplicates removed
- Records cleaned
- Distribution by geography

Example report:

```
======================================================================
LEADS DATABASE REPORT
======================================================================
Generated: 2026-02-01 12:00:00 UTC

STATISTICS:
  • Total leads: 500
  • Leads with phone: 485 (97.0%)
  • Leads with email: 275 (55.0%)
  • Leads with both: 270

IMPORT/EXPORT:
  • Total imported: 500
  • Duplicates removed: 23
  • Records cleaned: 487
  • Total exported: 500

DISTRIBUTION BY STATE (3 states):
  • TX: 350 leads
  • CA: 100 leads
  • FL: 50 leads

DISTRIBUTION BY CITY (15 cities):
  • Austin: 125 leads
  • Houston: 110 leads
  • Dallas: 85 leads
  ...
======================================================================
```

## 🔧 Advanced Configuration

### Custom Documents Directory

```python
extractor = DealMachineLeadsExtractor(
    documents_dir="/custom/path/to/leads"
)
```

### Headless vs Visible Browser

```python
# Headless (background)
extractor = DealMachineLeadsExtractor(headless=True)

# Visible (see what's happening)
extractor = DealMachineLeadsExtractor(headless=False)
```

### Extraction Limits

```python
# Extract up to 5000 leads
leads = await extractor.smart_extract(
    email=email,
    password=password,
    max_leads=5000
)
```

### Custom Filters

```python
# Only leads from Austin with both phone and email
austin_leads = manager.filter_leads(
    city="Austin",
    state="TX",
    has_phone=True,
    has_email=True
)

# Export just those
manager.export_csv("austin_complete_leads.csv", austin_leads)
```

## 🎨 Workflow Examples

### Example 1: Daily Lead Export

```bash
# Extract today's leads
python3 dealmachine_leads_extractor.py

# Organize and export
python3 csv_leads_manager.py
# Choose: Import -> Deduplicate -> Export by City
```

### Example 2: Merge Multiple Sources

```python
manager = CSVLeadsManager()

# Import from multiple exports
manager.import_csv("export_2026_01_01.csv")
manager.import_csv("export_2026_01_15.csv")
manager.import_csv("export_2026_02_01.csv")

# Remove duplicates
manager.deduplicate()  # Keeps latest version

# Export merged & clean data
manager.export_master_csv()
```

### Example 3: Geographic Targeting

```python
manager = CSVLeadsManager()
manager.import_csv("all_leads.csv")

# Get Texas leads only
tx_leads = manager.filter_leads(state="TX")

# Export by city within Texas
for city in ["Austin", "Dallas", "Houston", "San Antonio"]:
    city_leads = [l for l in tx_leads if l.city == city]
    if city_leads:
        manager.export_csv(f"tx_{city.lower()}_leads.csv", city_leads)
```

### Example 4: Complete Automated Workflow

```bash
# One command does everything
python3 run_complete_leads_workflow.py

# Enter credentials
# Choose max leads
# Walk away and let it work!

# Come back to:
# - Master CSV with all leads
# - Separate CSVs by state
# - Separate CSVs by city
# - Comprehensive report
```

## 🛠️ Troubleshooting

### "No leads found"
- **Check credentials**: Ensure email/password are correct
- **Check leads tab**: Make sure you have leads in DealMachine
- **Try headless=False**: See what the browser is doing
- **Check network**: Ensure you can reach dealmachine.com

### "Login failed"
- **Verify credentials**: Double-check email and password
- **Check 2FA**: Disable two-factor authentication if enabled
- **Try manual login**: Make sure you can login via browser

### "Browser not found"
```bash
# Install Playwright browsers
python3 -m playwright install chromium
```

### "Import failed"
- **Check file path**: Ensure CSV file exists
- **Check CSV format**: Ensure headers match expected fields
- **Check encoding**: Files should be UTF-8 encoded

### "No duplicates found" (but you know there are)
- Duplicates are matched by phone + address
- Different phone formats may not match
- Run `clean_data()` first to standardize

## 💡 Pro Tips

1. **Run demo first**: Test with sample data before real extraction
2. **Use headless mode**: Faster and uses less resources
3. **Deduplicate regularly**: Before each export
4. **Clean data first**: Before deduplication for better matching
5. **Export by geography**: Easier to work with smaller organized files
6. **Keep master backup**: Always have a master CSV with everything
7. **Track statistics**: Monitor DNC filter rate and data quality

## 🔒 Privacy & Security

- ✅ Credentials never stored (entered per session)
- ✅ Local processing only (no cloud uploads)
- ✅ DNC compliance built-in
- ✅ All data saved locally
- ✅ No external API calls (except DealMachine)

## 📝 Notes

- **DNC Filtering**: Relies on DealMachine's DNC data
- **Rate Limiting**: Respectful delays to avoid blocking
- **Browser Detection**: Uses stealth mode to avoid detection
- **Error Handling**: Gracefully handles network issues
- **Extensible**: Easy to add custom fields or logic

## 🚀 Next Steps

After extraction, you can:

1. **Import to CRM**: Upload CSVs to your CRM system
2. **Email campaigns**: Use emails for outreach
3. **Call campaigns**: Use phones for calling
4. **Direct mail**: Use addresses for mailers
5. **Further analysis**: Import to Excel/Google Sheets

## 📞 Support

For issues or questions:
- Check this README first
- Review error messages carefully
- Try demo mode to isolate issues
- Check DealMachine's site status

---

## 🎉 Success!

You now have the most powerful DealMachine leads extraction and organization system!

**What you can do:**
- ✅ Extract 1000s of clean leads automatically
- ✅ Filter out DNC leads instantly
- ✅ Organize by any criteria
- ✅ Export to perfectly formatted CSVs
- ✅ Merge multiple sources
- ✅ Track comprehensive statistics

**Your leads, organized, clean, and ready to use!** 🚀

---

*Built with ❤️ for real estate investors who demand the best*
