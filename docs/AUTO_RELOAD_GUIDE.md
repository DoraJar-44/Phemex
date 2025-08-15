# 🔄 Auto-Reload Development Guide

## Overview
The Phemex trading bot includes **instant auto-reload** functionality that automatically restarts the bot whenever you make ANY code changes. This eliminates the need to manually stop and restart the bot during development.

## 🚀 Quick Start

### For Scanner Mode (Original Bot)
```powershell
# Windows PowerShell
.\start_scanner_autoreload.ps1
```

### For Unified Bot Mode  
```powershell
# Windows PowerShell
.\start_unified_autoreload.ps1
```

### Manual Auto-Reload
```powershell
# Scanner mode
python auto_reload.py --mode scanner

# API server mode  
python auto_reload.py --mode api

# Unified bot mode
python auto_reload.py --mode unified
```

## ⚡ What Triggers Auto-Reload?

The auto-reload system monitors file changes and **instantly restarts** when you:

### ✅ **Always Triggers Restart:**
- **Edit any `.py` file** (even adding a space)
- **Modify `.env` files** (environment config)
- **Change `.mdc` files** (workspace rules)
- **Update `.json` files** (configuration)
- **Edit `.txt` files** (symbol lists, etc.)

### ❌ **Ignores These Files:**
- `.log` files (logs)
- `.pyc/.pyo` files (Python cache)
- `__pycache__/` directories
- Hidden files (starting with `.`)
- The `auto_reload.py` script itself (prevents loops)

## 🎛️ Configuration

### Restart Speed
- **Restart delay**: 0.5 seconds (super fast!)
- **Debounce protection**: Prevents rapid restarts from multiple file changes

### File Monitoring
- **Recursive monitoring**: Watches all subdirectories
- **Real-time detection**: Uses `watchdog` library for instant file change detection
- **Cross-platform**: Works on Windows, Linux, and macOS

## 🔧 How It Works

1. **File Watcher**: Monitors the entire project directory recursively
2. **Change Detection**: Detects file modifications, creations, and moves  
3. **Process Management**: Gracefully stops the current bot process
4. **Instant Restart**: Launches a new bot instance with updated code
5. **Output Streaming**: Shows bot output in real-time

## 💡 Development Workflow

### Traditional Workflow (Slow):
1. Edit code
2. Stop bot manually
3. Start bot manually  
4. Wait for initialization
5. Test changes
6. Repeat...

### Auto-Reload Workflow (Fast):
1. **Start auto-reload once**: `.\start_scanner_autoreload.ps1`
2. **Edit any code** and save
3. **Bot restarts automatically** (0.5s)
4. **Test immediately** 
5. **Repeat instantly** ⚡

## 🎯 Benefits

- **⚡ Lightning fast development cycle**
- **🔄 Zero manual restarts needed**
- **📝 Edit any file and see changes instantly**  
- **🎛️ Multiple bot modes supported**
- **💻 Works with any code editor**
- **🔧 Robust error handling**

## 🐛 Troubleshooting

### Bot Won't Restart
```bash
# Check if file is being monitored
echo "test" >> test_file.py
# Should trigger restart, then delete test_file.py
```

### Multiple Restarts
- The system has debounce protection (0.5s)
- Some editors save files multiple times - this is normal

### Process Not Stopping
- Auto-reload sends SIGTERM, then SIGKILL if needed
- Windows: Uses `terminate()` then `kill()`

### Missing Dependencies
```bash
pip install watchdog
```

## 🎨 Color Support

The auto-reload system maintains all the color support improvements:
- ✅ 256-color terminal support  
- ✅ Windows PowerShell compatibility
- ✅ Cursor/VSCode terminal integration
- ✅ Automatic terminal detection

## 📱 Example Usage

```powershell
PS> .\start_scanner_autoreload.ps1

🎯 Starting Phemex Scanner with INSTANT AUTO-RELOAD...
============================================================
✅ Python detected: Python 3.11.0
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

# ... TUI appears with colors ...
# Edit any file and save...

📝 File modified: scanner.py
🔄 Restarting bot with latest code...
🛑 Stopping bot (PID: 12345)...
🚀 Starting bot in scanner mode...
✅ Bot started with PID: 12346
```

## 🔥 Pro Tips

1. **Keep auto-reload running** during development sessions
2. **Use meaningful commit messages** since restart is so fast  
3. **Test small changes quickly** with instant feedback
4. **Edit multiple files** - system handles batch changes
5. **Use with version control** - see changes immediately

---

**Happy coding with instant auto-reload! 🚀⚡**
