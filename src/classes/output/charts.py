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
