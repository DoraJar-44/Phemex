#!/usr/bin/env python3
"""Simple clean display for the bot without TUI spam."""

import asyncio
import os
from bot.engine.scanner import run_scanner as original_run_scanner


async def run_clean_scanner():
    """Run scanner with simplified display."""
    print("🎯 Starting Phemex Trading Bot - Clean Mode")
    print("=" * 50)
    print("📊 Scanning symbols for trading signals...")
    print("⚡ Only showing scores ≥80 or active signals")
    print("🔄 Press Ctrl+C to stop")
    print("=" * 50)
    
    # Run the original scanner but with suppressed TUI
    os.environ["DASHBOARD"] = "false"  # Disable TUI completely
    await original_run_scanner()


if __name__ == "__main__":
    asyncio.run(run_clean_scanner())
