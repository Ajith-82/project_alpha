"""
Chart Builder

Unified charting interface for stock visualization.
Supports candlestick charts with technical indicators.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


logger = logging.getLogger(__name__)


@dataclass
class ChartConfig:
    """Chart configuration options."""
    width: int = 1200
    height: int = 800
    theme: str = "dark"  # dark, light
    show_volume: bool = True
    show_ma: bool = True
    show_macd: bool = False
    show_rsi: bool = False


class ChartBuilder:
    """
    Unified chart builder for stock visualization.
    
    Creates interactive Plotly charts with candlesticks,
    volume, and technical indicators.
    """
    
    def __init__(self, config: Optional[ChartConfig] = None):
        """
        Initialize chart builder.
        
        Args:
            config: Chart configuration
        """
        self.config = config or ChartConfig()
        self._fig: Optional[go.Figure] = None
        self._data: Optional[pd.DataFrame] = None
        self._ticker: str = ""
    
    def create(self, ticker: str, data: pd.DataFrame) -> "ChartBuilder":
        """
        Create a new chart for a stock.
        
        Args:
            ticker: Stock symbol
            data: Price data with OHLCV columns
            
        Returns:
            Self for chaining
        """
        if not PLOTLY_AVAILABLE:
            logger.warning("Plotly not available, charts disabled")
            return self
        
        self._ticker = ticker
        self._data = data.copy()
        
        # Determine subplot rows
        rows = 1
        row_heights = [0.7]
        
        if self.config.show_volume:
            rows += 1
            row_heights.append(0.15)
        if self.config.show_macd:
            rows += 1
            row_heights.append(0.15)
        if self.config.show_rsi:
            rows += 1
            row_heights.append(0.15)
        
        # Normalize heights
        total = sum(row_heights)
        row_heights = [h / total for h in row_heights]
        
        self._fig = make_subplots(
            rows=rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
        )
        
        # Add candlestick chart
        self._add_candlestick()
        
        # Add optional components
        current_row = 2
        if self.config.show_volume:
            self._add_volume(current_row)
            current_row += 1
        if self.config.show_macd and all(
            col in data.columns for col in ["MACD", "MACD_signal"]
        ):
            self._add_macd(current_row)
            current_row += 1
        if self.config.show_rsi and "RSI" in data.columns:
            self._add_rsi(current_row)
        
        # Apply theme
        self._apply_theme()
        
        return self
    
    def _add_candlestick(self):
        """Add candlestick chart."""
        self._fig.add_trace(
            go.Candlestick(
                x=self._data.index,
                open=self._data["Open"],
                high=self._data["High"],
                low=self._data["Low"],
                close=self._data["Close"],
                name="Price",
            ),
            row=1,
            col=1,
        )
        
        # Add moving averages
        if self.config.show_ma:
            for ma_col, color in [
                ("SMA_10", "yellow"),
                ("SMA_30", "orange"),
                ("SMA_50", "blue"),
                ("SMA_200", "red"),
            ]:
                if ma_col in self._data.columns:
                    self._fig.add_trace(
                        go.Scatter(
                            x=self._data.index,
                            y=self._data[ma_col],
                            name=ma_col,
                            line=dict(color=color, width=1),
                        ),
                        row=1,
                        col=1,
                    )
    
    def _add_volume(self, row: int):
        """Add volume bar chart."""
        colors = [
            "green" if close >= open_
            else "red"
            for open_, close in zip(
                self._data["Open"],
                self._data["Close"],
            )
        ]
        
        self._fig.add_trace(
            go.Bar(
                x=self._data.index,
                y=self._data["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.7,
            ),
            row=row,
            col=1,
        )
    
    def _add_macd(self, row: int):
        """Add MACD indicator."""
        self._fig.add_trace(
            go.Scatter(
                x=self._data.index,
                y=self._data["MACD"],
                name="MACD",
                line=dict(color="blue", width=1),
            ),
            row=row,
            col=1,
        )
        
        self._fig.add_trace(
            go.Scatter(
                x=self._data.index,
                y=self._data["MACD_signal"],
                name="Signal",
                line=dict(color="orange", width=1),
            ),
            row=row,
            col=1,
        )
        
        # Histogram
        if "MACD_hist" in self._data.columns:
            colors = [
                "green" if v >= 0 else "red"
                for v in self._data["MACD_hist"]
            ]
            self._fig.add_trace(
                go.Bar(
                    x=self._data.index,
                    y=self._data["MACD_hist"],
                    name="Histogram",
                    marker_color=colors,
                ),
                row=row,
                col=1,
            )
    
    def _add_rsi(self, row: int):
        """Add RSI indicator."""
        self._fig.add_trace(
            go.Scatter(
                x=self._data.index,
                y=self._data["RSI"],
                name="RSI",
                line=dict(color="purple", width=1),
            ),
            row=row,
            col=1,
        )
        
        # Add reference lines
        self._fig.add_hline(
            y=70, line_dash="dash", line_color="red",
            opacity=0.5, row=row, col=1,
        )
        self._fig.add_hline(
            y=30, line_dash="dash", line_color="green",
            opacity=0.5, row=row, col=1,
        )
    
    def _apply_theme(self):
        """Apply chart theme."""
        if self.config.theme == "dark":
            template = "plotly_dark"
            bgcolor = "#1e1e1e"
        else:
            template = "plotly_white"
            bgcolor = "#ffffff"
        
        self._fig.update_layout(
            title=f"{self._ticker} Stock Chart",
            template=template,
            paper_bgcolor=bgcolor,
            plot_bgcolor=bgcolor,
            xaxis_rangeslider_visible=False,
            height=self.config.height,
            width=self.config.width,
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )
    
    def add_indicator(
        self,
        name: str,
        values: pd.Series,
        row: int = 1,
        color: str = "cyan",
    ) -> "ChartBuilder":
        """
        Add a custom indicator line.
        
        Args:
            name: Indicator name
            values: Series of values
            row: Chart row (1-indexed)
            color: Line color
            
        Returns:
            Self for chaining
        """
        if self._fig is None:
            return self
        
        self._fig.add_trace(
            go.Scatter(
                x=values.index,
                y=values,
                name=name,
                line=dict(color=color, width=1),
            ),
            row=row,
            col=1,
        )
        return self
    
    def add_annotation(
        self,
        text: str,
        x: Any,
        y: float,
        color: str = "white",
    ) -> "ChartBuilder":
        """
        Add a text annotation.
        
        Args:
            text: Annotation text
            x: X position (date/index)
            y: Y position (price)
            color: Text color
            
        Returns:
            Self for chaining
        """
        if self._fig is None:
            return self
        
        self._fig.add_annotation(
            x=x,
            y=y,
            text=text,
            showarrow=True,
            arrowhead=2,
            font=dict(color=color),
        )
        return self
    
    def save(
        self,
        path: str,
        format: Optional[str] = None,
    ) -> str:
        """
        Save chart to file.
        
        Args:
            path: Output file path
            format: Override format (html, png, svg)
            
        Returns:
            Path to saved file
        """
        if self._fig is None:
            logger.warning("No chart to save")
            return ""
        
        # Detect format
        if format is None:
            ext = Path(path).suffix.lower()
            format = ext[1:] if ext else "html"
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        if format == "html":
            self._fig.write_html(path)
        elif format in ("png", "jpg", "jpeg", "svg", "pdf"):
            self._fig.write_image(path)
        else:
            self._fig.write_html(path)
        
        logger.info(f"Chart saved to {path}")
        return path
    
    def show(self):
        """Display chart in browser."""
        if self._fig:
            self._fig.show()
    
    def get_figure(self) -> Optional[go.Figure]:
        """Get the Plotly figure object."""
        return self._fig


# Convenience function
def create_stock_chart(
    ticker: str,
    data: pd.DataFrame,
    output_path: Optional[str] = None,
    show_volume: bool = True,
    show_macd: bool = False,
    show_rsi: bool = False,
) -> Optional[str]:
    """
    Create a stock chart.
    
    Args:
        ticker: Stock symbol
        data: Price data
        output_path: Save path (optional)
        show_volume: Include volume
        show_macd: Include MACD
        show_rsi: Include RSI
        
    Returns:
        Path to saved file or None
    """
    config = ChartConfig(
        show_volume=show_volume,
        show_macd=show_macd,
        show_rsi=show_rsi,
    )
    
    builder = ChartBuilder(config)
    builder.create(ticker, data)
    
    if output_path:
        return builder.save(output_path)
    else:
        builder.show()
        return None


def create_batch_charts(
    screener_name: str,
    market: str,
    symbols: List[str],
    data: Dict[str, Any],
    output_dir: str,
    batch_size: int = 100,
    send_email_flag: bool = True,
) -> List[str]:
    """
    Create charts for multiple symbols in batches.
    
    This is the modular replacement for create_plot_and_email_batched.
    
    Args:
        screener_name: Name of the screener (for email subject)
        market: Market identifier ('us' or 'india')
        symbols: List of stock symbols
        data: Dictionary containing 'price_data' with DataFrames per symbol
        output_dir: Base output directory
        batch_size: Number of charts per batch
        send_email_flag: Whether to send email notifications
        
    Returns:
        List of paths to saved chart files
    """
    import math
    import os
    
    saved_files = []
    price_data = data.get("price_data", data)
    
    num_batches = math.ceil(len(symbols) / batch_size)
    
    for batch_idx in range(num_batches):
        batch_symbols = symbols[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        batch_output_dir = f"{output_dir}_batch_{batch_idx}"
        os.makedirs(batch_output_dir, exist_ok=True)
        
        for idx, symbol in enumerate(batch_symbols, start=1):
            if symbol not in price_data:
                logger.warning(f"No data for {symbol}, skipping")
                continue
            
            df = price_data[symbol]
            if df is None or df.empty:
                continue
            
            # Use last 200 bars
            df = df.tail(200).copy()
            
            # Create chart with all indicators
            chart_path = _create_full_chart(
                ticker=symbol,
                data=df,
                output_path=f"{batch_output_dir}/{idx}_{symbol}.svg",
                market=market,
            )
            
            if chart_path:
                saved_files.append(chart_path)
                
        # Prepare data for email report
        if send_email_flag:
            try:
                from classes.output.email import EmailConfig, EmailServer
                config_path = "email_config.json"
                if os.path.exists(config_path):
                    config = EmailConfig.from_json(config_path)
                    server = EmailServer(config)
                    
                    # Collect summary data for ALL symbols
                    summary_data = []
                    for symbol in batch_symbols:
                        if symbol not in price_data: continue
                        df = price_data[symbol]
                        if df is None or df.empty: continue
                        
                        # Calculate change
                        last_close = df["Close"].iloc[-1]
                        prev_close = df["Close"].iloc[-2]
                        change_pct = ((last_close - prev_close) / prev_close) * 100
                        
                        # Get Sector (if available in data["sectors"])
                        sector = data.get("sectors", {}).get(symbol, "N/A")
                        
                        summary_data.append({
                            "symbol": symbol,
                            "price": f"{last_close:.2f}",
                            "change": change_pct,
                            "sector": sector
                        })

                    # Generate PNGs only for top 10 for embedding
                    chart_files = []
                    top_symbols = batch_symbols[:10]
                    
                    for symbol in top_symbols:
                        if symbol not in price_data: continue
                        df = price_data[symbol]
                        if df is None or df.empty: continue

                        # Generate PNG for embedding (Plotly write_image requires kaleido)
                        png_path = f"{batch_output_dir}/{symbol}.png"
                        try:
                            _create_full_chart(symbol, df.tail(200), png_path, market)
                            if os.path.exists(png_path):
                                chart_files.append(png_path)
                        except Exception as e:
                            logger.warning(f"Failed to create PNG for {symbol}: {e}")

                    # Send the rich report
                    if summary_data:
                        # Sort summary data by change % descending for the report
                        summary_data.sort(key=lambda x: x['change'], reverse=True)
                        
                        server.send_stock_report_email(
                            subject=f"{screener_name} - {market.upper()} Report (Batch {batch_idx + 1})",
                            market=market,
                            category=screener_name,
                            summary_data=summary_data,
                            charts=chart_files,
                            mock=False # Set to False in production
                        )
                        
            except FileNotFoundError:
                logger.debug("Email config not found, skipping email")
            except Exception as e:
                logger.warning(f"Email failed: {e}")
    
    logger.info(f"Created {len(saved_files)} charts in {output_dir}")
    return saved_files


def _create_full_chart(
    ticker: str,
    data: pd.DataFrame,
    output_path: str,
    market: str = "us",
) -> Optional[str]:
    """
    Create a full stock chart with all indicators (SMA, Donchian, Volume, MACD, RSI).
    
    Args:
        ticker: Stock symbol
        data: Price data DataFrame
        output_path: Path to save the chart
        market: Market identifier
        
    Returns:
        Path to saved file or None
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available")
        return None
    
    try:
        import plotly.io as pio
        
        # Get TradingView recommendation if available
        recommendation_str = ""
        try:
            import classes.Tools as tools
            recommendation = tools.tradingview_recommendation(ticker, market)
            recommendation_str = ' '.join(recommendation)
        except Exception:
            pass
        
        # Build date breaks for non-trading days
        dt_all = pd.date_range(start=data.index[0], end=data.index[-1])
        dt_obs = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(data.index)]
        dt_breaks = [d for d in dt_all.strftime("%Y-%m-%d").tolist() if d not in dt_obs]
        
        # Create 4-row subplot
        fig = make_subplots(
            rows=4,
            cols=1,
            vertical_spacing=0.01,
            shared_xaxes=True,
            row_heights=(2, 0.5, 1, 1),
        )
        
        # Row 1: Price with SMAs
        for col in ["Close", "SMA_10", "SMA_30", "SMA_50", "SMA_200"]:
            if col in data.columns:
                fig.add_trace(
                    go.Scatter(x=data.index, y=data[col], name=col),
                    row=1, col=1
                )
        
        # Add Donchian channels
        _add_donchian_to_chart(fig, data)
        
        # Row 2: Volume
        colors = [
            "#9C1F0B" if row["Open"] - row["Close"] >= 0 else "#2B8308"
            for _, row in data.iterrows()
        ]
        fig.add_trace(
            go.Bar(x=data.index, y=data["Volume"], showlegend=False, marker_color=colors),
            row=2, col=1
        )
        
        # Row 3: MACD
        if all(col in data.columns for col in ["MACD", "MACD_signal", "MACD_hist"]):
            macd_colors = ["green" if v >= 0 else "red" for v in data["MACD_hist"]]
            fig.add_trace(
                go.Bar(x=data.index, y=data["MACD_hist"], showlegend=False, marker_color=macd_colors),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data["MACD"], name="MACD", line=dict(color="black", width=2)),
                row=3, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data["MACD_signal"], name="Signal", line=dict(color="blue", width=1)),
                row=3, col=1
            )
        
        # Row 4: RSI
        if "RSI" in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data["RSI"], name="RSI", line=dict(color="purple", width=2)),
                row=4, col=1
            )
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=4, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=4, col=1)
            fig.update_yaxes(range=[0, 100], row=4, col=1)
        
        # Y-axis labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="MACD", row=3, col=1)
        fig.update_yaxes(title_text="RSI", row=4, col=1)
        
        # Remove weekends/holidays
        fig.update_xaxes(tickangle=45, rangebreaks=[dict(values=dt_breaks)])
        
        # Layout
        title_suffix = f" - {recommendation_str}" if recommendation_str else ""
        fig.update_layout(
            template="seaborn",
            title=f"{ticker}{title_suffix}",
            height=900,
            width=1200,
            legend_title="Legend",
            margin=dict(r=10, b=10),
        )
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Infer format from extension (e.g. .png -> png)
        ext = Path(output_path).suffix.lower()
        fmt = ext[1:] if ext else "svg"
        pio.write_image(fig, output_path, format=fmt)
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating chart for {ticker}: {e}")
        return None


def _add_donchian_to_chart(fig: go.Figure, data: pd.DataFrame, window: int = 20):
    """Add Donchian channel bands to chart."""
    try:
        from ta.volatility import DonchianChannel
        donchian = DonchianChannel(high=data["High"], low=data["Low"], close=data["Close"], window=window)
        don_high = donchian.donchian_channel_hband()
        don_mid = donchian.donchian_channel_mband()
        don_low = donchian.donchian_channel_lband()
    except ImportError:
        don_high = data["High"].rolling(window=window).max()
        don_low = data["Low"].rolling(window=window).min()
        don_mid = (don_high + don_low) / 2
    
    fig.add_trace(
        go.Scatter(x=data.index, y=don_high, name="Don High", line=dict(color="red", width=2, dash="dash")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=don_mid, name="Don Mid", line=dict(color="blue", width=2, dash="dash")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data.index, y=don_low, name="Don Low", line=dict(color="purple", width=2, dash="dash")),
        row=1, col=1
    )

