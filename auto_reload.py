#!/usr/bin/env python3
"""Auto-reloader for the Phemex trading bot using watchdog."""

import os
import sys
import time
import subprocess
import signal
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class BotReloadHandler(FileSystemEventHandler):
    """Handle file changes and restart the bot."""
    
    def __init__(self, script_path: str, mode: str = "scanner"):
        self.script_path = script_path
        self.mode = mode
        self.process = None
        self.restart_needed = False
        self.last_restart = 0
        self.restart_delay = 2  # seconds
        
        # Start the bot initially
        self.start_bot()
    
    def start_bot(self):
        """Start the bot process."""
        if self.process:
            self.stop_bot()
        
        print(f"🚀 Starting bot in {self.mode} mode...")
        
        # Set environment variables
        env = os.environ.copy()
        env["MODE"] = self.mode
        
        try:
            if self.mode == "api":
                # Use uvicorn for API mode
                self.process = subprocess.Popen(
                    [sys.executable, "-m", "uvicorn", "bot.api.server:app", "--host", "0.0.0.0", "--port", "8000"],
                    env=env,
                    cwd=Path(__file__).parent,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            else:
                # Use main.py for scanner mode
                self.process = subprocess.Popen(
                    [sys.executable, self.script_path],
                    env=env,
                    cwd=Path(__file__).parent,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            print(f"✅ Bot started with PID: {self.process.pid}")
            
        except Exception as e:
            print(f"❌ Failed to start bot: {e}")
    
    def stop_bot(self):
        """Stop the bot process."""
        if self.process:
            print(f"🛑 Stopping bot (PID: {self.process.pid})...")
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("⚠️ Force killing bot process...")
                self.process.kill()
                self.process.wait()
            except Exception as e:
                print(f"⚠️ Error stopping bot: {e}")
            
            self.process = None
    
    def should_restart(self, event_path: str) -> bool:
        """Check if we should restart for this file change."""
        path = Path(event_path)
        
        # Only restart for Python files
        if path.suffix != '.py':
            return False
        
        # Ignore __pycache__ and other temp files
        if '__pycache__' in str(path) or path.name.startswith('.'):
            return False
        
        # Ignore this auto-reload script to prevent loops
        if path.name == 'auto_reload.py':
            return False
        
        return True
    
    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory and self.should_restart(event.src_path):
            current_time = time.time()
            
            # Debounce rapid file changes
            if current_time - self.last_restart < self.restart_delay:
                return
            
            self.last_restart = current_time
            file_name = Path(event.src_path).name
            print(f"📝 File changed: {file_name}")
            print("🔄 Restarting bot with latest code...")
            
            self.start_bot()


def main():
    """Main auto-reload function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Auto-reload Phemex trading bot")
    parser.add_argument("--mode", choices=["scanner", "api"], default="scanner",
                       help="Bot mode to run (default: scanner)")
    parser.add_argument("--script", default="main.py",
                       help="Script to run (default: main.py)")
    
    args = parser.parse_args()
    
    # Get the project root directory
    project_root = Path(__file__).parent
    script_path = project_root / args.script
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        sys.exit(1)
    
    print("🎯 Phemex Bot Auto-Reloader")
    print("=" * 40)
    print(f"📂 Watching: {project_root}")
    print(f"🎮 Mode: {args.mode}")
    print(f"📜 Script: {args.script}")
    print("🔄 Will auto-restart on .py file changes")
    print("⏹️ Press Ctrl+C to stop")
    print("=" * 40)
    
    # Create event handler and observer
    event_handler = BotReloadHandler(str(script_path), args.mode)
    observer = Observer()
    
    # Watch the entire project directory
    observer.schedule(event_handler, str(project_root), recursive=True)
    
    try:
        observer.start()
        
        # Keep the main thread alive and show bot output
        while True:
            if event_handler.process:
                try:
                    # Read and display bot output
                    output = event_handler.process.stdout.readline()
                    if output:
                        print(output.strip())
                    elif event_handler.process.poll() is not None:
                        # Process ended, restart it
                        print("⚠️ Bot process ended unexpectedly, restarting...")
                        event_handler.start_bot()
                except Exception as e:
                    print(f"⚠️ Error reading bot output: {e}")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down auto-reloader...")
        event_handler.stop_bot()
        observer.stop()
    
    observer.join()
    print("✅ Auto-reloader stopped")


if __name__ == "__main__":
    main()
