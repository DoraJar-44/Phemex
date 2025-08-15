#!/usr/bin/env python3
"""Setup script for live trading configuration."""

import os
import shutil
from pathlib import Path


def setup_env_file():
    """Create .env file from template if it doesn't exist."""
    env_file = Path(".env")
    template_file = Path(".env.template")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return
    
    if not template_file.exists():
        print("❌ .env.template not found")
        return
    
    shutil.copy(template_file, env_file)
    print("✅ Created .env file from template")
    print("⚠️  IMPORTANT: Edit .env file with your actual API credentials!")


def check_requirements():
    """Check if all required packages are installed."""
    try:
        import fastapi, uvicorn, ccxt, httpx, structlog
        print("✅ All required packages installed")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Run: python -m pip install -r requirements.txt")


def test_api_connection():
    """Test Phemex API connection."""
    try:
        from bot.config import settings
        if not settings.phemex_api_key or not settings.phemex_api_secret:
            print("⚠️  API credentials not set in .env file")
            return
        
        print("✅ API credentials loaded")
        print(f"Base URL: {settings.phemex_base_url}")
        print(f"Live trade: {settings.live_trade}")
        
    except Exception as e:
        print(f"❌ API configuration error: {e}")


def main():
    """Main setup function."""
    print("🚀 Phemex Live Trading Setup")
    print("=" * 40)
    
    setup_env_file()
    check_requirements()
    test_api_connection()
    
    print("\n📋 Next Steps:")
    print("1. Edit .env file with your Phemex API credentials")
    print("2. Set LIVE_TRADE=false for testing, true for live trading")
    print("3. Adjust ACCOUNT_BALANCE_USDT to your actual balance")
    print("4. Run: python scripts/phemex_balance_check.py")
    print("5. Run: MODE=scanner python main.py")


if __name__ == "__main__":
    main()
