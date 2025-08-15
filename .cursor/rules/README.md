# 🎯 Cursor Rules for Phemex Trading Bot

This directory contains comprehensive Cursor rules that **STRICTLY ENFORCE** the proper usage, structure, and safety of your Phemex trading bot.

## 📋 Rule Files Overview

### 1. **main.mdc** - Core Global Rules
- **Always Applied**: Yes (global enforcement)
- **Purpose**: Main trading bot rules and execution flow
- **Scope**: All Python, Markdown, JSON, and environment files

### 2. **api_safety.mdc** - API Security Rules
- **Always Applied**: Yes (critical security)
- **Purpose**: Protect API keys and enforce secure API usage
- **Scope**: All Python files and environment files

### 3. **file_structure.mdc** - File Organization Rules
- **Always Applied**: Yes (structure protection)
- **Purpose**: Maintain proper file structure and prevent unauthorized changes
- **Scope**: All files and directories

### 4. **trading_safety.mdc** - Risk Management Rules
- **Always Applied**: Yes (trading safety)
- **Purpose**: Enforce risk limits and safe trading practices
- **Scope**: All Python files and environment files

## 🚨 Critical Enforcement Areas

### **API Key Protection**
- Never log, display, or commit API keys
- Always load from environment variables
- Implement proper authentication patterns

### **File Structure Protection**
- Never modify core bot files without permission
- Maintain exact directory structure
- Prevent unauthorized file creation

### **Trading Safety**
- Never enable live trading without explicit permission
- Always enforce risk limits
- Always use reduce-only orders
- Always validate before trading

### **Code Quality**
- Use proper type hints
- Implement comprehensive error handling
- Follow Python best practices

## 🔧 How Rules Work Together

1. **main.mdc** provides the foundation and execution flow
2. **api_safety.mdc** ensures secure API usage
3. **file_structure.mdc** maintains code organization
4. **trading_safety.mdc** protects against trading risks

## 📁 Protected Structure

```
NEW-PHEMEX-main/
├── .cursor/                    # These rules (NEVER DELETE)
├── bot/                       # Core bot package (NEVER MODIFY)
│   ├── __init__.py
│   ├── config.py
│   ├── api/
│   ├── engine/
│   ├── exchange/
│   ├── execution/
│   ├── risk/
│   ├── signals/
│   ├── strategy/
│   ├── ui/
│   ├── utils/
│   └── validation/
├── main.py                    # Main entry (NEVER DELETE)
├── unified_trading_bot.py     # Core bot (NEVER DELETE)
├── setup_live_trading.py      # Setup (NEVER DELETE)
├── env.template               # Template (NEVER DELETE)
├── requirements.txt            # Dependencies (NEVER DELETE)
└── README.md                  # Documentation (NEVER DELETE)
```

## ⚠️ Rule Violation Response

If any rule is violated:
1. **IMMEDIATELY STOP** the operation
2. **REPORT** the violation to you
3. **REQUEST** explicit permission
4. **CORRECT** any unauthorized changes
5. **VERIFY** compliance before proceeding

## 🔒 What These Rules Prevent

- **Security Breaches**: API key exposure, unauthorized access
- **Structure Damage**: File deletion, wrong locations, duplicates
- **Trading Risks**: Live trading without permission, risk limit violations
- **Code Issues**: Poor quality, missing validation, unsafe patterns

## ✅ Compliance Benefits

- **Consistent Code**: All files follow the same patterns
- **Secure Trading**: No accidental live trading or risk violations
- **Maintainable Structure**: Clear organization and dependencies
- **Professional Quality**: Type hints, error handling, documentation

## 🚀 Getting Started

1. **Review** all rule files to understand requirements
2. **Test** your bot with LIVE_TRADE=false first
3. **Verify** all risk parameters are correct
4. **Monitor** compliance during development
5. **Report** any rule violations immediately

## 📞 Rule Enforcement

These rules are **automatically enforced** by Cursor. They will:
- Prevent unauthorized file changes
- Block unsafe trading operations
- Enforce secure API usage
- Maintain code quality standards

**Remember**: These rules exist to protect your trading bot, your capital, and your security. Always follow them strictly!
