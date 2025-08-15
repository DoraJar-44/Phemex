# 🎯 Phemex Trading Bot

Advanced automated trading bot for Phemex Futures with real-time scanning, signal detection, and auto-reload capabilities.

## ✨ Features

### 🔄 **Auto-Reload System**
- **Watchdog-powered** file monitoring
- **Automatic restarts** on code changes
- **Real-time development** workflow
- **Clean process management**

### 📊 **Trading Engine**
- **Real-time scanning** of 286+ symbols
- **Signal detection** with scoring system
- **Bracket orders** (Entry + TP1/TP2 + Stop Loss)
- **Risk management** with position sizing
- **Dry run and live trading** modes

### 🎛️ **User Interface**
- **3x3 Grid TUI** with live data
- **Console output** filtering (only high scores)
- **Clean display** with anti-spam features
- **Real-time validation** status

### ⚡ **Performance**
- **Optimized scanning** (2-second intervals)
- **Smart filtering** (scores ≥80 for console, ≥75 for TUI)
- **Resource management** with proper cleanup
- **Error handling** and recovery

## 🚀 Quick Start

### **Option 1: Auto-Reload Mode (Recommended)**
```powershell
python run_with_reload.py
```

### **Option 2: API Mode with Auto-Reload**
```powershell
$env:MODE="api"
python run_with_reload.py
```

### **Option 3: Windows Scripts**
```powershell
# Scanner mode
.\start_bot.ps1 scanner

# API mode  
.\start_bot.ps1 api
```

### **Option 4: Traditional Mode**
```powershell
# Scanner mode
python main.py

# API mode
$env:MODE="api"
python main.py
```

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Stressica1/NEW-PHEMEX.git
   cd NEW-PHEMEX
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   # Copy and edit the environment template
   copy env.template .env
   # Edit .env with your API keys and settings
   ```

## ⚙️ Configuration

### **Environment Variables**
```env
# Phemex API
PHEMEX_API_KEY=your_api_key
PHEMEX_SECRET=your_secret
PHEMEX_BASE_URL=https://api.phemex.com

# Trading Settings
LIVE_TRADE=false
RISK_PER_TRADE_PCT=1.0
ACCOUNT_BALANCE_USDT=1000
LEVERAGE_MAX=20

# Scanner Settings
SYMBOLS_FILTER=50x
DASHBOARD=true
USE_TUI=true
```

### **Trading Modes**
- **`scanner`**: Continuous market scanning for signals
- **`api`**: Webhook receiver for TradingView alerts

## 📊 TUI Dashboard

The 3x3 grid displays:

| **🎯 High Scores** | **📊 Live Feed** | **⚡ Signals** |
|---|---|---|
| **📈 Long Analysis** | **🎛️ System Status** | **📉 Short Analysis** |
| **⚠️ Validation** | **💰 Trades** | **📋 Stats** |

## 🛡️ Safety Features

- **Reduce-only stops** (never increase position)
- **One-way mode** (`posSide: Merged`)
- **Position awareness** (50% size reduction when adding)
- **Validation checks** before execution
- **Dry run mode** for testing

## 🔧 Development

### **Auto-Reload Development**
```bash
# Edit any .py file → Bot automatically restarts
# 2-second debounce prevents spam
# Clean process management
```

### **Manual Restart**
```bash
# Press Ctrl+C to stop
# Run again to restart with changes
```

## 📈 Performance Metrics

- **Symbols**: 286+ monitored
- **Scan Rate**: Every 2 seconds
- **Response Time**: <100ms
- **Memory**: Optimized with cleanup
- **Uptime**: Auto-recovery on errors

## 🎯 Trading Strategy

### **Signal Scoring**
- **Long/Short scores** (0-100)
- **Threshold**: ≥80 for execution
- **Multiple timeframes** analysis
- **Volume and momentum** factors

### **Risk Management**
- **Position sizing** based on account risk
- **Stop-loss** at signal levels
- **Take-profit** levels (TP1: 50%, TP2: 25%)
- **Maximum leverage** limits

## 🔗 API Integration

### **TradingView Webhooks**
```json
POST /webhook/tradingview
{
  "action": "BUY",
  "symbol": "BTCUSDT",
  "price": 45000,
  "levels": {
    "stop": 44000,
    "tp1": 46000,
    "tp2": 47000
  }
}
```

## 📝 Logs and Monitoring

- **Real-time console** output
- **TUI dashboard** updates
- **Error logging** with context
- **Performance metrics** tracking

## 🤝 Support

For issues, feature requests, or questions:
- Create an issue on GitHub
- Check the logs for error details
- Ensure environment variables are set

## 📄 License

This project is for educational and personal use. Please comply with Phemex Terms of Service and local regulations.

---

**⚡ Always running the latest code with auto-reload! 🎉**
