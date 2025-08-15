import asyncio
import ccxt.async_support as ccxt

async def test_phemex_timeframes():
    ex = ccxt.phemex({'enableRateLimit': True, 'options': {'defaultType':'swap','defaultSubType':'linear'}})
    await ex.load_markets()
    
    test_symbol = 'BTC/USDT:USDT'
    timeframes = ['1m', '3m', '5m', '15m', '30m', '1h', '4h', '1d']
    valid_timeframes = []
    
    print(f'Testing timeframes with {test_symbol}:')
    
    for tf in timeframes:
        try:
            data = await ex.fetch_ohlcv(test_symbol, timeframe=tf, limit=10)
            if data and len(data) >= 5:
                valid_timeframes.append(tf)
                print(f'✅ {tf:4s} - OK ({len(data)} bars)')
            else:
                print(f'❌ {tf:4s} - Insufficient data')
        except Exception as e:
            print(f'❌ {tf:4s} - Error: {str(e)[:50]}')
    
    await ex.close()
    print(f'\nValid timeframes: {valid_timeframes}')
    return valid_timeframes

if __name__ == "__main__":
    asyncio.run(test_phemex_timeframes())
