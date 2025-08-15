#!/usr/bin/env python3
import os
import sys
import time
import hmac
import hashlib
import json
import urllib.parse
import urllib.request


def get_env(name: str, default: str = "") -> str:
	val = os.getenv(name, default)
	if not val:
		raise SystemExit(f"Missing required env var: {name}")
	return val


def build_signature(secret: str, path: str, query: str, expiry: str, body: str) -> str:
	message = f"{path}{query}{expiry}{body}"
	return hmac.new(secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256).hexdigest()


def http_get(url: str, headers: dict) -> dict:
	req = urllib.request.Request(url, headers=headers, method="GET")
	with urllib.request.urlopen(req, timeout=20) as resp:
		data = resp.read()
		return json.loads(data.decode("utf-8"))


def main() -> None:
	# Inputs
	api_key = get_env("PHEMEX_API_KEY")
	api_secret = get_env("PHEMEX_API_SECRET")
	base_url = os.getenv("PHEMEX_BASE_URL", "https://api.phemex.com")
	currency = sys.argv[1] if len(sys.argv) > 1 else os.getenv("PHEMEX_CURRENCY", "USDT")

	# Endpoint
	path = "/accounts/accountInfo"
	query = f"currency={urllib.parse.quote(currency)}"
	expiry = str(int(time.time()) + 60)
	body = ""

	# Signature
	signature = build_signature(api_secret, path, query, expiry, body)

	# Request
	url = f"{base_url}{path}?{query}"
	headers = {
		"x-phemex-access-token": api_key,
		"x-phemex-request-expiry": expiry,
		"x-phemex-request-signature": signature,
	}

	try:
		resp_json = http_get(url, headers)
		print(json.dumps(resp_json, indent=2, sort_keys=True))
	except urllib.error.HTTPError as e:
		print(f"HTTPError: {e.code}")
		try:
			print(e.read().decode("utf-8"))
		except Exception:
			pass
		raise SystemExit(1)
	except Exception as e:
		print(f"Error: {e}")
		raise SystemExit(1)


if __name__ == "__main__":
	main()
