# 🔧 Windows 11 KB5065426 Network Share Fix

## Problem

After Windows 11 update KB5065426, WSL network shares (`\\wsl$\` and `\\wsl.localhost\`) no longer work.

## Solutions

### ✅ **Solution 1: Use Dual-Save Script** (Recommended)

This saves files to BOTH WSL and Windows locations automatically:

```bash
cd ~/agents/examples/fullstack-rag-video-platform/scraper
source ~/venv-leads/bin/activate
python3 run_dual_save_workflow.py
```

Files will be saved to:
- **WSL**: `~/Documents/dealmachine_data/leads/`
- **Windows**: `C:\Users\YOUR_USERNAME\Documents\dealmachine_data\leads\`

Access from Windows: Just open File Explorer and go to your Documents folder!

---

### 🔄 **Solution 2: Uninstall KB5065426**

If you need WSL network shares working:

1. Open **Settings** → **Windows Update** → **Update History**
2. Scroll down and click **Uninstall updates**
3. Find **KB5065426**
4. Click **Uninstall**
5. Restart your computer

**Note**: This removes the update but you'll lose any security fixes it provided.

---

### 📁 **Solution 3: Use /mnt/c/ Path Directly**

Access Windows files from WSL using `/mnt/c/`:

```bash
# Save directly to Windows Documents
mkdir -p /mnt/c/Users/YOUR_USERNAME/Documents/dealmachine_data/leads

# Then run extractor with Windows path
python3 dealmachine_leads_extractor.py
# When prompted, use: /mnt/c/Users/YOUR_USERNAME/Documents/dealmachine_data/leads
```

---

### 🛠️ **Solution 4: Registry Fix (Advanced)**

Microsoft may release a registry fix. Check for updates:

1. Open **Settings** → **Windows Update**
2. Click **Check for updates**
3. Install any new updates that fix the issue

---

### 💡 **Solution 5: Map Network Drive**

Try mapping WSL as a network drive:

1. Open File Explorer
2. Right-click **This PC** → **Map network drive**
3. Choose a drive letter (e.g., `Z:`)
4. Folder: `\\wsl.localhost\Debian\home\YOUR_USERNAME`
5. Check **Reconnect at sign-in**
6. Click **Finish**

---

## Quick Test

Check if WSL shares work:

```powershell
# From PowerShell
dir \\wsl.localhost\Debian\home
```

If you get "Access Denied" or "Network path not found", use Solution 1 (Dual-Save).

---

## File Locations

### Using Dual-Save Script:

**WSL Location:**
```
~/Documents/dealmachine_data/leads/
```

**Windows Location:**
```
C:\Users\YOUR_USERNAME\Documents\dealmachine_data\leads\
```

### Using /mnt/c/ Path:

**From WSL:**
```
/mnt/c/Users/YOUR_USERNAME/Documents/dealmachine_data/leads/
```

**From Windows:**
```
C:\Users\YOUR_USERNAME\Documents\dealmachine_data\leads\
```

---

## Recommended Approach

**For most users**: Use **Solution 1** (Dual-Save Script)

This automatically:
- ✅ Saves to WSL (fast, works with all tools)
- ✅ Copies to Windows (accessible from File Explorer)
- ✅ No need to uninstall updates
- ✅ No registry changes needed

**Run it:**
```bash
cd ~/agents/examples/fullstack-rag-video-platform/scraper
source ~/venv-leads/bin/activate
python3 run_dual_save_workflow.py
```

Choose option 2 for demo mode to test!

---

## Verification

After running dual-save script, verify files exist:

**From WSL:**
```bash
ls -la ~/Documents/dealmachine_data/leads/demo_organized/
```

**From Windows:**
1. Open File Explorer
2. Navigate to: `C:\Users\YOUR_USERNAME\Documents\dealmachine_data\leads\demo_organized\`
3. You should see your CSV files!

---

## Alternative: Direct Save to Windows

Modify any script to save directly to `/mnt/c/`:

```python
# Instead of:
documents_dir = "~/Documents/dealmachine_data/leads"

# Use:
documents_dir = "/mnt/c/Users/YOUR_USERNAME/Documents/dealmachine_data/leads"
```

---

## Need Help?

If none of these work:

1. Check your Windows username: Open CMD and type `echo %USERNAME%`
2. Verify `/mnt/c/Users/` exists: Run `ls /mnt/c/Users/` in WSL
3. Try saving to a different location like Desktop: `/mnt/c/Users/YOUR_USERNAME/Desktop/leads/`

---

**Bottom Line**: Use `run_dual_save_workflow.py` and your files will be accessible from both WSL and Windows! 🎉
