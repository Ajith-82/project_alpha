"""
Console utilities for Rich-based CLI interface.

Provides progress bars, formatted tables, panels, and styling for the
Project Alpha stock screening application.
"""

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.style import Style
from rich import box
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

# Global console instance
console = Console()

# Color scheme
COLORS = {
    "primary": "cyan",
    "success": "green",
    "warning": "yellow", 
    "error": "red",
    "info": "blue",
    "muted": "dim white",
}


def print_banner():
    """Display the application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║   ██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗  ║
    ║   ██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝  ║
    ║   ██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║     ║
    ║   ██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║     ║
    ║   ██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║     ║
    ║   ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝     ║
    ║                     █████╗ ██╗     ██████╗ ██╗  ██╗ █████╗    ║
    ║                    ██╔══██╗██║     ██╔══██╗██║  ██║██╔══██╗   ║
    ║                    ███████║██║     ██████╔╝███████║███████║   ║
    ║                    ██╔══██║██║     ██╔═══╝ ██╔══██║██╔══██║   ║
    ║                    ██║  ██║███████╗██║     ██║  ██║██║  ██║   ║
    ║                    ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝   ║
    ║                                                               ║
    ║          Your Day-to-Day Trading Companion                    ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def print_section(title: str, icon: str = "📊"):
    """Print a section header."""
    console.print()
    console.rule(f"[bold cyan]{icon} {title}[/bold cyan]", style="cyan")
    console.print()


def print_success(message: str):
    """Print a success message."""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str):
    """Print an error message."""
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str):
    """Print a warning message."""
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_info(message: str):
    """Print an info message."""
    console.print(f"[blue]ℹ[/blue] {message}")


@contextmanager
def create_progress(description: str = "Processing..."):
    """Create a progress bar context manager."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        yield progress


def create_download_progress():
    """Create a progress bar specifically for downloads."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
    )


def create_results_table(title: str, columns: List[str]) -> Table:
    """Create a styled results table."""
    table = Table(
        title=title,
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="cyan",
        show_lines=True,
    )
    for col in columns:
        table.add_column(col, justify="center")
    return table


def create_stock_table(title: str, stocks: List[Dict[str, Any]]) -> Table:
    """Create a table for displaying stock results."""
    table = Table(
        title=f"[bold]{title}[/bold]",
        box=box.ROUNDED,
        header_style="bold white on blue",
        border_style="blue",
        padding=(0, 1),
    )
    
    table.add_column("Symbol", style="cyan bold", justify="left")
    table.add_column("Signal", justify="center")
    table.add_column("Price", justify="right")
    table.add_column("Change", justify="right")
    
    for stock in stocks:
        symbol = stock.get("symbol", "N/A")
        signal = stock.get("signal", "—")
        price = stock.get("price", "—")
        change = stock.get("change", 0)
        
        # Color change based on value
        if isinstance(change, (int, float)):
            if change > 0:
                change_str = f"[green]+{change:.2f}%[/green]"
            elif change < 0:
                change_str = f"[red]{change:.2f}%[/red]"
            else:
                change_str = f"[white]{change:.2f}%[/white]"
        else:
            change_str = str(change)
        
        # Color signal
        if signal.upper() in ["BUY", "BULLISH", "STRONG UP"]:
            signal_str = f"[green bold]{signal}[/green bold]"
        elif signal.upper() in ["SELL", "BEARISH", "STRONG DOWN"]:
            signal_str = f"[red bold]{signal}[/red bold]"
        else:
            signal_str = f"[yellow]{signal}[/yellow]"
        
        table.add_row(symbol, signal_str, str(price), change_str)
    
    return table


def print_summary_panel(market: str, total_stocks: int, screeners_run: List[str], results: Dict[str, int]):
    """Print a summary panel at the end of execution."""
    
    # Build summary content
    content = []
    content.append(f"[cyan]Market:[/cyan] {market.upper()}")
    content.append(f"[cyan]Stocks Analyzed:[/cyan] {total_stocks}")
    content.append("")
    content.append("[bold]Screener Results:[/bold]")
    
    for screener, count in results.items():
        if count > 0:
            content.append(f"  • {screener}: [green]{count} matches[/green]")
        else:
            content.append(f"  • {screener}: [dim]{count} matches[/dim]")
    
    panel = Panel(
        "\n".join(content),
        title="[bold white]📈 Screening Summary[/bold white]",
        border_style="green",
        padding=(1, 2),
    )
    console.print()
    console.print(panel)


def print_config_panel(args):
    """Print configuration panel showing current settings."""
    content = []
    content.append(f"[cyan]Market:[/cyan] {args.market.upper()}")
    content.append(f"[cyan]Ranking:[/cyan] {args.rank}")
    content.append(f"[cyan]Cache:[/cyan] {'Enabled' if args.cache else 'Disabled'}")
    content.append(f"[cyan]Plots:[/cyan] {'Disabled' if args.no_plots else 'Enabled'}")
    
    if args.db_path:
        content.append(f"[cyan]Database:[/cyan] {args.db_path}")
    if args.load_model:
        content.append(f"[cyan]Load Model:[/cyan] {args.load_model}")
    if args.save_model:
        content.append(f"[cyan]Save Model:[/cyan] {args.save_model}")
    
    panel = Panel(
        "\n".join(content),
        title="[bold white]⚙️ Configuration[/bold white]",
        border_style="cyan",
        padding=(1, 2),
    )
    console.print(panel)
    console.print()


def format_price(price: float) -> str:
    """Format price with color based on value."""
    return f"${price:,.2f}"


def format_change(change: float) -> str:
    """Format percentage change with color."""
    if change > 0:
        return f"[green]+{change:.2f}%[/green]"
    elif change < 0:
        return f"[red]{change:.2f}%[/red]"
    return f"{change:.2f}%"
