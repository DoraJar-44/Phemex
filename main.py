import os
import asyncio


async def _maybe_run_scanner():
	from bot.engine.scanner import run_scanner
	print("Starting native scanner (no webhooks)...")
	await run_scanner()


def _run_api():
	try:
		import uvicorn
	except ImportError:
		raise SystemExit("uvicorn is not installed. Install dependencies first: pip install -r requirements.txt")
	host = os.getenv("HOST", "127.0.0.1")
	port = int(os.getenv("PORT", "8000"))
	reload = os.getenv("RELOAD", "true").lower() in ("1", "true", "yes")
	uvicorn.run("bot.api.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
	mode = os.getenv("MODE", "scanner")  # scanner | api
	if mode == "api":
		_run_api()
	else:
		asyncio.run(_maybe_run_scanner())
