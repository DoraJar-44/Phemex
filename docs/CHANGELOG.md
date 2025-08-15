# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-08-15

### Fixed
- **Scoring System**: Resolved critical scoring calculation issues where most inputs were defaulting to 0
  - Fixed `bounce_prob` calculation based on price proximity to support/resistance levels
  - Fixed `bias_conf` calculation based on trend strength and signal quality
  - Enhanced scoring inputs in both `bot/api/server.py` and `bot/engine/scanner.py`
- **Score Filtering**: Added missing score validation in webhook handler to reject signals below minimum threshold
- **Score Requirements**: Corrected scoring system to properly meet the 85-point minimum requirement
  - Previous system could only achieve maximum of 80 points due to missing inputs
  - Now achieves up to 150 points with proper input calculations

### Added
- **Enhanced Webhook Handler**: Implemented comprehensive score filtering in `/webhook/tradingview` endpoint
  - Signals below minimum score (85) are now rejected with detailed logging
  - Added proper bounce probability calculation based on price-to-level proximity
  - Added bias confidence scoring based on signal strength and trend analysis
- **Improved Scanner Scoring**: Enhanced the scanner's scoring algorithm with:
  - Dynamic bounce probability calculation for both long and short positions
  - Trend confidence scoring based on SMA position and strength
  - Separate scoring inputs for long and short setups
- **Live Trading Controls**: Enhanced live trading safety features
  - Score filtering is now enforced before any trade execution
  - Dry-run mode properly indicated in webhook responses
  - Detailed rejection logging with score breakdown

### Changed
- **Scoring Algorithm**: Completely overhauled the scoring system to use all available inputs:
  - Base Score: 50 points (unchanged)
  - Range Score: 0-30 points based on price position relative to support/resistance
  - Bounce Score: 0-20 points based on calculated bounce probability (0-0.9 scale)
  - Divergence Score: 0-20 points (framework ready for future implementation)
  - Bias Score: 0-30 points based on trend confidence and signal strength
- **Webhook Response Format**: Enhanced webhook responses to include:
  - Score value in all responses
  - Rejection reason when signals fail score requirements
  - Dry-run indication when live trading is disabled

### Technical Details
- **Bounce Probability Calculation**:
  - For longs: `max(0, 0.9 - (distance_to_support / range_size))`
  - For shorts: `max(0, 0.9 - (distance_to_resistance / range_size))`
- **Bias Confidence Scoring**:
  - Strong signals: 0.7 confidence
  - Regular signals: 0.4 confidence
  - Trend-based scaling: `min(0.8, trend_strength * 10)`
- **Score Validation**: Minimum 85 points required (configurable via `SCORE_MIN` environment variable)

### Server Status
- **Live Trading Server**: Successfully running on `http://localhost:8000`
- **Webhook Endpoint**: Active at `http://localhost:8000/webhook/tradingview`
- **Dashboard**: Accessible at `http://localhost:8000/static/index.html`
- **Health Check**: Available at `http://localhost:8000/health`

---

## [1.0.0] - Previous Release

### Initial Features
- Predictive Ranges (PR) strategy implementation
- ATR-based support/resistance calculation
- Basic scoring system
- Bracket order execution
- Risk management and position sizing
- TradingView webhook integration
- Scanner mode for automated discovery
- Dashboard and monitoring interface
