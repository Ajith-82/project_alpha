#!/usr/bin/env python
"""
Project Alpha - Stock Market Screening Application

A comprehensive stock screening tool with volatility prediction,
technical analysis, and multiple screening strategies.

Examples:
    # Analyze US market with all screeners
    python project_alpha.py --market us

    # Run only volatility and trend screeners on India market
    python project_alpha.py --market india --screeners volatility,trend

    # Get top 20 results in JSON format
    python project_alpha.py --market us --top 20 --format json

    # Filter stocks by price range
    python project_alpha.py --market us --min-price 10 --max-price 500
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "4"

import sys
import click
import rich_click as click

# Configure rich-click styling
click.rich_click.USE_RICH_MARKUP = True
click.rich_click.USE_MARKDOWN = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.STYLE_OPTION = "bold cyan"
click.rich_click.STYLE_SWITCH = "bold green"
click.rich_click.STYLE_METAVAR = "yellow"
click.rich_click.STYLE_USAGE = "bold"

# Define option groups for organized help
click.rich_click.OPTION_GROUPS = {
    "project_alpha.py": [
        {
            "name": "Market Selection",
            "options": ["--market", "--symbols"],
        },
        {
            "name": "Screener Configuration", 
            "options": ["--screeners", "--rank", "--top", "--min-price", "--max-price"],
        },
        {
            "name": "Output Options",
            "options": ["--format", "--save-table", "--no-plots", "--plot-losses", "--verbose", "--quiet", "--no-banner"],
        },
        {
            "name": "Data & Caching",
            "options": ["--cache", "--no-cache", "--db-path"],
        },
        {
            "name": "Model Options",
            "options": ["--load-model", "--save-model"],
        },
        {
            "name": "Additional Features",
            "options": ["--value"],
        },
    ],
}

from classes.Download import load_data, load_volatile_data
from classes.Volatile import volatile
from classes.screeners import BreakoutScreener, TrendlineScreener
from classes.output import (
    create_batch_charts,
    console, print_banner, print_section, print_success, print_error,
    print_warning, print_info, print_summary_panel, print_config_panel,
    create_download_progress
)
import classes.IndexListFetcher as Index
import classes.Tools as tools


# Available screeners
AVAILABLE_SCREENERS = ["all", "volatility", "breakout", "trend", "ma", "macd", "donchain"]


def validate_screeners(ctx, param, value):
    """Validate and parse screener selection."""
    if not value:
        return ["all"]
    screeners = [s.strip().lower() for s in value.split(",")]
    for s in screeners:
        if s not in AVAILABLE_SCREENERS:
            raise click.BadParameter(
                f"Invalid screener '{s}'. Choose from: {', '.join(AVAILABLE_SCREENERS)}"
            )
    return screeners


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-m", "--market",
    type=click.Choice(["us", "india"], case_sensitive=False),
    default="us",
    show_default=True,
    help="Market to analyze. **us** = S&P 500, **india** = NSE 500",
)
@click.option(
    "-s", "--symbols",
    type=str,
    multiple=True,
    hidden=True,
    help="Specific stock symbols to analyze (overrides market selection)",
)
@click.option(
    "--screeners",
    type=str,
    default="all",
    show_default=True,
    callback=validate_screeners,
    help="Comma-separated screeners to run: **all**, volatility, breakout, trend, ma, macd, donchain",
)
@click.option(
    "-r", "--rank",
    type=click.Choice(["rate", "growth", "volatility"], case_sensitive=False),
    default="growth",
    show_default=True,
    help="Result ranking method. **rate** = trend position, **growth** = trend strength, **volatility** = volatility estimate",
)
@click.option(
    "-t", "--top",
    type=int,
    default=None,
    metavar="N",
    help="Limit results to top N stocks per screener",
)
@click.option(
    "--min-price",
    type=float,
    default=None,
    metavar="PRICE",
    help="Minimum stock price filter (e.g., 10.00)",
)
@click.option(
    "--max-price",
    type=float,
    default=None,
    metavar="PRICE",
    help="Maximum stock price filter (e.g., 500.00)",
)
@click.option(
    "-f", "--format",
    "output_format",
    type=click.Choice(["table", "csv", "json"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format for results",
)
@click.option(
    "--save-table/--no-save-table",
    default=True,
    show_default=True,
    help="Save results to CSV file",
)
@click.option(
    "--no-plots",
    is_flag=True,
    default=False,
    help="Disable chart generation",
)
@click.option(
    "--plot-losses",
    is_flag=True,
    default=False,
    help="Show loss function decay during model training",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    default=False,
    help="Enable verbose output with debug information",
)
@click.option(
    "-q", "--quiet",
    is_flag=True,
    default=False,
    help="Minimal output (errors only)",
)
@click.option(
    "--no-banner",
    is_flag=True,
    default=False,
    help="Skip the ASCII banner",
)
@click.option(
    "--cache/--no-cache",
    default=True,
    show_default=True,
    help="Use cached data if available",
)
@click.option(
    "--db-path",
    type=click.Path(),
    default=None,
    metavar="PATH",
    help="SQLite database path for persistent price data storage",
)
@click.option(
    "--load-model",
    type=click.Path(exists=True),
    default=None,
    metavar="FILE",
    help="Load pre-trained model parameters from pickle file",
)
@click.option(
    "--save-model",
    type=click.Path(),
    default=None,
    metavar="FILE",
    help="Save trained model parameters to pickle file",
)
@click.option(
    "--value",
    is_flag=True,
    default=False,
    help="Include value stocks from external screener sources (India only)",
)
@click.version_option(version="0.1.0", prog_name="Project Alpha")
def cli(market, symbols, screeners, rank, top, min_price, max_price, output_format,
        save_table, no_plots, plot_losses, verbose, quiet, no_banner, cache,
        db_path, load_model, save_model, value):
    """
    🚀 **Project Alpha** - Your Day-to-Day Trading Companion
    
    A comprehensive stock screening tool featuring:
    
    - **Volatility Analysis**: TensorFlow Probability-based predictions
    - **Technical Screeners**: Breakout, Trend, MA, MACD, Donchain
    - **Multi-Market Support**: US (S&P 500) and India (NSE 500)
    
    ---
    
    **Quick Start:**
    
        $ python project_alpha.py --market us
        $ python project_alpha.py --market india --screeners volatility,trend
    """
    # Create args namespace for backward compatibility
    class Args:
        pass
    
    args = Args()
    args.market = market.lower()
    args.symbols = list(symbols) if symbols else None
    args.screeners = screeners
    args.rank = rank.lower()
    args.top = top
    args.min_price = min_price
    args.max_price = max_price
    args.output_format = output_format.lower()
    args.save_table = save_table
    args.no_plots = no_plots
    args.plot_losses = plot_losses
    args.verbose = verbose
    args.quiet = quiet
    args.no_banner = no_banner
    args.cache = cache
    args.db_path = db_path
    args.load_model = load_model
    args.save_model = save_model
    args.value = value
    
    # Run main with enhanced args
    run_screening(args)


def screener_value_charts(cache, market: str, index: str, symbols: list, db_path: str = None):
    """Generate value stock charts for a given market and symbols."""
    historic_data_dir = f"data/historic_data/{market}"
    os.makedirs(historic_data_dir, exist_ok=True)
    
    file_prefix = f"{index}_data"
    data = load_data(cache, symbols, market, file_prefix, historic_data_dir, db_path=db_path)
    
    price_data = data["price_data"]
    value_symbols = data["tickers"]
    
    processed_data_dir = f"data/processed_data/{index}"
    os.makedirs(processed_data_dir, exist_ok=True)
    
    create_batch_charts("IND_screener_value_stocks", market, value_symbols, {"price_data": price_data}, processed_data_dir)


def run_screening(args):
    """Run the stock screening application with the given arguments."""
    
    # Display banner (unless --no-banner or --quiet)
    if not args.no_banner and not args.quiet:
        print_banner()
    
    # Display configuration
    if not args.quiet:
        print_config_panel(args)
    
    # Initialize variables
    cache = args.cache
    market = args.market
    db_path = args.db_path
    results = {}  # Track screener results
    screeners_to_run = args.screeners if hasattr(args, 'screeners') else ["all"]
    
    # Cleanup report directories
    tools.cleanup_directory_files("data/processed_data")
    if not args.quiet:
        print_success("Cleaned up previous report directories")
    
    # Load market data
    if not args.quiet:
        print_section("Loading Market Data", "📥")
    
    if market == "india":
        index, symbols = Index.nse_500()
        screener_dur = 3
        if not args.quiet:
            print_info(f"Loading NSE 500 index ({len(symbols)} stocks)")
        if args.value:
            value_index, value_symbols = Index.ind_screener_value_stocks()
            screener_value_charts(cache, market, value_index, value_symbols, db_path)
    else:
        index, symbols = Index.sp_500()
        screener_dur = 3
        if not args.quiet:
            print_info(f"Loading S&P 500 index ({len(symbols)} stocks)")

    # Create data directories
    data_dir = f"data/historic_data/{market}"
    os.makedirs(data_dir, exist_ok=True)
    
    file_prefix = f"{index}_data"
    data = load_data(cache, symbols, market, file_prefix, data_dir, db_path=db_path)
    
    total_stocks = len(data.get("tickers", []))
    if not args.quiet:
        print_success(f"Loaded data for {total_stocks} stocks")
    
    # Apply price filters if specified
    if hasattr(args, 'min_price') and args.min_price is not None:
        if not args.quiet:
            print_info(f"Filtering stocks with price >= ${args.min_price}")
    if hasattr(args, 'max_price') and args.max_price is not None:
        if not args.quiet:
            print_info(f"Filtering stocks with price <= ${args.max_price}")
    
    # Determine which screeners to run
    run_all = "all" in screeners_to_run
    
    # Volatility Screening
    if run_all or "volatility" in screeners_to_run:
        if not args.quiet:
            print_section("Volatility Screening", "📊")
            console.print("[dim]Running TensorFlow Probability model...[/dim]")
        
        volatile_data = load_volatile_data(market, data)
        volatile_df = volatile(args, volatile_data)
        volatile_symbols_top = volatile_df["SYMBOL"].head(200).tolist()
        volatile_symbols_bottom = volatile_df["SYMBOL"].tail(200).tolist()
        
        if not args.quiet:
            print_success(f"Identified {len(volatile_symbols_top)} high-volatility stocks")
            print_success(f"Identified {len(volatile_symbols_bottom)} low-volatility stocks")
        
        # Categorize stocks by trading signal interpretation
        # 1. TREND (Momentum): High GROWTH + High VOLATILITY (top growth stocks)
        # 2. VALUE (Mean-reversion): BELOW TREND or HIGHLY BELOW TREND (undervalued)
        # 3. BREAKOUT: Low VOLATILITY + ALONG TREND (consolidating, ready to break)
        
        trend_candidates = volatile_df[
            (volatile_df["GROWTH"] > 0.001) & 
            (volatile_df["VOLATILITY"] > 0.10)
        ]["SYMBOL"].head(50).tolist()
        
        value_candidates = volatile_df[
            volatile_df["RATE"].isin(["BELOW TREND", "HIGHLY BELOW TREND"])
        ]["SYMBOL"].head(50).tolist()
        
        breakout_candidates = volatile_df[
            (volatile_df["RATE"] == "ALONG TREND") & 
            (volatile_df["VOLATILITY"] < 0.15) &
            (volatile_df["GROWTH"].abs() < 0.001)
        ]["SYMBOL"].head(50).tolist()
        
        if not args.quiet:
            print_info(f"Trading Categories: Trend={len(trend_candidates)}, Value={len(value_candidates)}, Breakout={len(breakout_candidates)}")
        
        # Generate plots for each trading category
        if not args.no_plots:
            # Trend/Momentum trading candidates
            if trend_candidates:
                trend_dir = "data/processed_data/volatile_trend_trading"
                if not args.quiet:
                    print_info(f"Generating {len(trend_candidates)} Trend/Momentum charts...")
                create_batch_charts("Trend Trading", market, trend_candidates, data, trend_dir)
            
            # Value/Mean-reversion trading candidates
            if value_candidates:
                value_dir = "data/processed_data/volatile_value_trading"
                if not args.quiet:
                    print_info(f"Generating {len(value_candidates)} Value/Undervalued charts...")
                create_batch_charts("Value Trading", market, value_candidates, data, value_dir)
            
            # Breakout trading candidates
            if breakout_candidates:
                breakout_dir = "data/processed_data/volatile_breakout_trading"
                if not args.quiet:
                    print_info(f"Generating {len(breakout_candidates)} Breakout charts...")
                create_batch_charts("Breakout Trading", market, breakout_candidates, data, breakout_dir)
        
        results["Volatility"] = len(volatile_symbols_top)
        results["Trend_Candidates"] = len(trend_candidates)
        results["Value_Candidates"] = len(value_candidates)
        results["Breakout_Candidates"] = len(breakout_candidates)
    else:
        volatile_symbols_top = []
        volatile_symbols_bottom = []
    
    # Breakout Screening
    if run_all or "breakout" in screeners_to_run:
        if not args.quiet:
            print_section("Breakout Screening", "🚀")
        
        breakout_screener_out_dir = "data/processed_data/screener_breakout"
        tickers_to_screen = volatile_symbols_bottom if volatile_symbols_bottom else data.get("tickers", [])
        
        # Use new modular screener
        breakout_screener = BreakoutScreener()
        price_data = data.get("price_data", {})
        batch_result = breakout_screener.screen_batch(tickers_to_screen, price_data)
        breakout_screener_out_symbols = [r.ticker for r in batch_result.buys]
        
        # Apply --top limit if specified
        if hasattr(args, 'top') and args.top and len(breakout_screener_out_symbols) > args.top:
            breakout_screener_out_symbols = breakout_screener_out_symbols[:args.top]
        
        if breakout_screener_out_symbols:
            if not args.no_plots:
                create_batch_charts("Breakout screener", market, breakout_screener_out_symbols, data, breakout_screener_out_dir)
            if args.save_table:
                tools.save_screener_results_to_csv(market, "screener_breakout", breakout_screener_out_symbols)
            if not args.quiet:
                print_success(f"Found {len(breakout_screener_out_symbols)} breakout candidates")
        else:
            if not args.quiet:
                print_warning("No breakout candidates found")
        results["Breakout"] = len(breakout_screener_out_symbols)
    
    # Trend Screening
    if run_all or "trend" in screeners_to_run:
        if not args.quiet:
            print_section("Trend Screening", "📈")
        
        trend_screener_out_dir = "data/processed_data/screener_trend"
        trend_screener_history = "data/processed_data/screener_trend_history"
        
        tickers_to_screen = volatile_symbols_top if volatile_symbols_top else data.get("tickers", [])[:200]
        
        # Use new modular screener
        trend_screener = TrendlineScreener(lookback_days=screener_dur)
        price_data = data.get("price_data", {})
        batch_result = trend_screener.screen_batch(tickers_to_screen, price_data)
        trend_screener_out_symbols = [r.ticker for r in batch_result.buys]
        
        # Apply --top limit if specified
        if hasattr(args, 'top') and args.top and len(trend_screener_out_symbols) > args.top:
            trend_screener_out_symbols = trend_screener_out_symbols[:args.top]
        
        if trend_screener_out_symbols:
            if not args.no_plots:
                create_batch_charts("Trend screener", market, trend_screener_out_symbols, data, trend_screener_out_dir)
            if args.save_table:
                trend_history_file = tools.save_screener_results_to_csv(market, "screener_trend", trend_screener_out_symbols)
            if not args.quiet:
                print_success(f"Found {len(trend_screener_out_symbols)} trending stocks")
            
            # Check for consistently trending stocks
            if args.save_table:
                trend_common = tools.find_common_symbols(trend_history_file, 5)
                if len(trend_common) > 0:
                    if not args.no_plots:
                        create_batch_charts("Trending stocks in last 5 days", market, trend_common, data, trend_screener_history)
                    if not args.quiet:
                        print_success(f"Found {len(trend_common)} consistently trending stocks")
        else:
            if not args.quiet:
                print_warning("No trending stocks found")
        results["Trend"] = len(trend_screener_out_symbols)
    
    # Summary
    if not args.quiet:
        screeners_run = [s for s in ["Volatility", "Breakout", "Trend"] if s in results]
        print_summary_panel(
            market=market,
            total_stocks=total_stocks,
            screeners_run=screeners_run,
            results=results
        )
        
        console.print("\n[dim]Reports saved to data/processed_data/[/dim]")
        console.print("[bold green]✨ Screening complete![/bold green]\n")


if __name__ == "__main__":
    cli()
