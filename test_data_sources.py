#!/usr/bin/env python
"""
Test script for multi-source data fetching system.
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from classes.DataSourceManager import get_manager, reset_manager
from classes.DataSourceConfig import get_config


def test_configuration():
    """Test configuration loading."""
    print("=" * 60)
    print("TESTING CONFIGURATION")
    print("=" * 60)

    config = get_config()
    print(f"\nConfiguration: {config}")
    print(f"\nAvailable sources: {config.get_available_sources()}")

    print("\nConfiguration validation:")
    status = config.validate_configuration()
    for source, msg in status.items():
        print(f"  {source}: {msg}")

    print("\n" + "=" * 60 + "\n")


def test_data_fetch():
    """Test data fetching with fallback."""
    print("=" * 60)
    print("TESTING DATA FETCH")
    print("=" * 60)

    # Reset manager for clean test
    reset_manager()

    # Get manager with verbose output
    manager = get_manager(verbose=True)

    # Test with a few tickers
    test_tickers = [
        ("us", "AAPL", "Apple"),
        ("us", "MSFT", "Microsoft"),
        ("us", "GOOGL", "Google"),
    ]

    print("\nTesting data fetch for sample tickers...\n")

    results = {}
    for market, ticker, name in test_tickers:
        print(f"\n--- Fetching {name} ({ticker}) ---")
        price_data, company_info = manager.fetch_stock_data(
            market=market,
            ticker=ticker,
            start_date="2024-01-01",
            end_date="2024-12-31",
            interval="1d"
        )

        if price_data is not None:
            results[ticker] = {
                "success": True,
                "records": len(price_data),
                "date_range": f"{price_data.index[0]} to {price_data.index[-1]}"
            }
            print(f"  Success! Retrieved {len(price_data)} records")
        else:
            results[ticker] = {"success": False}
            print(f"  Failed to retrieve data")

    # Print results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for ticker, result in results.items():
        if result["success"]:
            print(f"✓ {ticker}: {result['records']} records ({result['date_range']})")
        else:
            print(f"✗ {ticker}: Failed")

    # Print statistics
    manager.print_statistics()


def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# MULTI-SOURCE DATA FETCHER TEST SUITE")
    print("#" * 60 + "\n")

    try:
        # Test 1: Configuration
        test_configuration()

        # Test 2: Data fetching
        test_data_fetch()

        print("\n" + "#" * 60)
        print("# ALL TESTS COMPLETED")
        print("#" * 60 + "\n")

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
