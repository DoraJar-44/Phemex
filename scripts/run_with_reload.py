#!/usr/bin/env python3
"""Simple wrapper to run the bot with auto-reload capabilities."""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """Run the bot with auto-reload based on MODE environment variable."""
    
    # Get mode from environment or default to scanner
    mode = os.environ.get("MODE", "scanner").lower()
    
    print("🎯 Phemex Bot with Auto-Reload")
    print("=" * 40)
    print(f"🎮 Mode: {mode}")
    print("🔄 Auto-restart on code changes")
    print("⏹️ Press Ctrl+C to stop")
    print("=" * 40)
    
    # Run the auto-reloader
    try:
        subprocess.run([
            sys.executable, 
            "auto_reload.py", 
            "--mode", mode
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Stopping bot...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Auto-reloader failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
