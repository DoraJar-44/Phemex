# 🚀 How to Start the Bot with Auto-Reload

## PowerShell Execution Policy Issues? Here are ALL the solutions:

### ✅ **Method 1: Python Script (RECOMMENDED - No Policy Issues)**
```bash
python start_scanner.py
```
**Benefits:** 
- ✅ No execution policy issues
- ✅ Works on any system
- ✅ Full auto-reload functionality
- ✅ Clean output

### ✅ **Method 2: Batch File (Windows Alternative)**
```bash
start_scanner_autoreload.bat
```
**Benefits:**
- ✅ No execution policy issues  
- ✅ Native Windows batch
- ✅ Double-click to run

### ✅ **Method 3: PowerShell with Bypass (If you prefer PS)**
```powershell
powershell -ExecutionPolicy Bypass -File start_scanner_autoreload.ps1
```

### ✅ **Method 4: Direct Auto-Reload Command**
```bash
python auto_reload.py --mode scanner
```

### ✅ **Method 5: Enable PowerShell Scripts (One-time setup)**
```powershell
# Run as Administrator once:
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
# Then you can run:
.\start_scanner_autoreload.ps1
```

## 🎯 **EASIEST METHOD (Recommended):**

Just run this single command:
```bash
python start_scanner.py
```

This will:
- ✅ Set up test environment automatically
- ✅ Start the scanner with full color TUI
- ✅ Enable instant auto-reload (0.5s restart)
- ✅ Work without any PowerShell issues
- ✅ Monitor ALL code files for changes

## 🔄 **Test the Auto-Reload:**

1. Run: `python start_scanner.py`
2. Wait for the colorful TUI to appear
3. Open any `.py` file in the project
4. Add a space somewhere and save (Ctrl+S)
5. Watch the bot restart instantly! ⚡

## 🎨 **Full Feature Set:**

- **⚡ 0.5 second restart time**
- **🎨 Full color TUI support** 
- **📝 Monitors all code files**
- **🔄 Instant restart on ANY change**
- **💻 Works with any editor**
- **🛡️ Robust error handling**

## 📱 **What You'll See:**

```
============================================================
🎯 Starting Phemex Scanner with INSTANT AUTO-RELOAD...
============================================================
✅ Environment configured for testing (dry run mode)
🔄 INSTANT AUTO-RELOAD ENABLED!
📝 The bot will restart automatically when you:
   - Edit ANY .py file
   - Modify .env files
   - Change .mdc or .json config files
   - Even just add a space and save!

⚡ Restart delay: 0.5 seconds (super fast!)
⏹️ Press Ctrl+C to stop the auto-reloader
============================================================
🎯 Phemex Bot Auto-Reloader
========================================
📂 Watching: C:\Users\user\Desktop\NEW-PHEMEX-main
🎮 Mode: scanner
🔄 Will auto-restart on .py file changes
⏹️ Press Ctrl+C to stop
========================================
🚀 Starting bot in scanner mode...
✅ Bot started with PID: 12345
Starting TUI... Press 'q' to quit TUI interface.

[Beautiful colored TUI appears with live data]

# Edit any file and save...
📝 File modified: scanner.py
🔄 Restarting bot with latest code...
🛑 Stopping bot (PID: 12345)...
🚀 Starting bot in scanner mode...
✅ Bot started with PID: 12346
```

## 🏆 **Bottom Line:**

**Use this command and you're done:**
```bash
python start_scanner.py
```

No more PowerShell issues, no more manual restarts, just pure development bliss! 🚀
