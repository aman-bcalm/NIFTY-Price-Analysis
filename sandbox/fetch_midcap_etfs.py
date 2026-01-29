"""Probe: fetch recent data for Midcap 150 ETFs to pick a working ticker."""
import sys
from pathlib import Path

import yfinance as yf

# Midcap 150 ETFs (Yahoo ticker, expense ratio for reference)
ETFS = [
    ("MID150BEES.NS", 0.21),
    ("MIDCAPETF.NS", 0.05),
    ("MIDCAPIETF.NS", 0.15),
    ("HDFCMID150.NS", 0.20),
    ("MID150CASE.NS", 0.21),
]

def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    try:
        yf.set_tz_cache_location(str(root / "data" / "cache"))
    except Exception:
        pass

    print("Fetching last 5 trading days for each Midcap 150 ETF...\n")
    working = []
    for ticker, er in ETFS:
        try:
            df = yf.download(
                ticker,
                start="2026-01-01",
                end="2026-01-30",
                interval="1d",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df is not None and not df.empty:
                n = len(df)
                last = df.index.max()
                working.append((ticker, er, n, last))
                print(f"  OK  {ticker}  (ER {er}%)  rows={n}  last={last.date()}")
            else:
                print(f"  --  {ticker}  (ER {er}%)  no data")
        except Exception as e:
            print(f"  FAIL {ticker}  (ER {er}%)  {e}")

    if working:
        # Prefer lowest expense ratio
        best = min(working, key=lambda x: x[1])
        print(f"\nRecommended: {best[0]} (expense ratio {best[1]}%)")
    else:
        print("\nNo ETF returned data.")
    return 0 if working else 1

if __name__ == "__main__":
    raise SystemExit(main())
