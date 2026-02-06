"""
Result Formatters

Provides formatters for converting screening results to different output formats.
"""

import csv
import json
from abc import ABC, abstractmethod
from io import StringIO
from typing import Any, Dict, List, Optional

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ResultFormatter(ABC):
    """Abstract base class for result formatters."""
    
    @abstractmethod
    def format(self, results: List[Dict[str, Any]]) -> str:
        """
        Format results to string output.
        
        Args:
            results: List of result dictionaries
            
        Returns:
            Formatted string
        """
        pass
    
    @abstractmethod
    def get_content_type(self) -> str:
        """Get MIME content type for this format."""
        pass


class CSVFormatter(ResultFormatter):
    """Format results as CSV."""
    
    def __init__(self, delimiter: str = ",", include_header: bool = True):
        """
        Initialize CSV formatter.
        
        Args:
            delimiter: Field delimiter
            include_header: Include header row
        """
        self.delimiter = delimiter
        self.include_header = include_header
    
    def format(self, results: List[Dict[str, Any]]) -> str:
        """Format results as CSV string."""
        if not results:
            return ""
        
        output = StringIO()
        
        # Get all keys from first result
        fieldnames = list(results[0].keys())
        
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            delimiter=self.delimiter,
            extrasaction="ignore",
        )
        
        if self.include_header:
            writer.writeheader()
        
        for result in results:
            writer.writerow(result)
        
        return output.getvalue()
    
    def get_content_type(self) -> str:
        return "text/csv"


class JSONFormatter(ResultFormatter):
    """Format results as JSON."""
    
    def __init__(self, indent: Optional[int] = 2, compact: bool = False):
        """
        Initialize JSON formatter.
        
        Args:
            indent: Indentation level (None for compact)
            compact: Use compact output
        """
        self.indent = None if compact else indent
    
    def format(self, results: List[Dict[str, Any]]) -> str:
        """Format results as JSON string."""
        return json.dumps(results, indent=self.indent, default=str)
    
    def get_content_type(self) -> str:
        return "application/json"


class TableFormatter(ResultFormatter):
    """Format results as Rich table (returns string representation)."""
    
    def __init__(self, title: Optional[str] = None, show_lines: bool = True):
        """
        Initialize table formatter.
        
        Args:
            title: Table title
            show_lines: Show row lines
        """
        self.title = title
        self.show_lines = show_lines
    
    def format(self, results: List[Dict[str, Any]]) -> str:
        """Format results as table string."""
        if not results:
            return "No results"
        
        if RICH_AVAILABLE:
            return self._format_rich(results)
        else:
            return self._format_plain(results)
    
    def _format_rich(self, results: List[Dict[str, Any]]) -> str:
        """Format using Rich table."""
        console = Console(record=True, width=120)
        
        table = Table(
            title=self.title,
            show_lines=self.show_lines,
            header_style="bold cyan",
        )
        
        # Add columns
        for key in results[0].keys():
            table.add_column(key.upper())
        
        # Add rows
        for result in results:
            table.add_row(*[str(v) for v in result.values()])
        
        console.print(table)
        return console.export_text()
    
    def _format_plain(self, results: List[Dict[str, Any]]) -> str:
        """Format as plain text table."""
        if not results:
            return ""
        
        headers = list(results[0].keys())
        
        # Calculate column widths
        widths = {h: len(h) for h in headers}
        for result in results:
            for h in headers:
                widths[h] = max(widths[h], len(str(result.get(h, ""))))
        
        # Build table
        lines = []
        
        # Header
        header_line = " | ".join(h.upper().ljust(widths[h]) for h in headers)
        lines.append(header_line)
        lines.append("-" * len(header_line))
        
        # Rows
        for result in results:
            row = " | ".join(str(result.get(h, "")).ljust(widths[h]) for h in headers)
            lines.append(row)
        
        return "\n".join(lines)
    
    def get_content_type(self) -> str:
        return "text/plain"


class HTMLFormatter(ResultFormatter):
    """Format results as HTML table."""
    
    def __init__(self, title: Optional[str] = None, css_class: str = "results-table"):
        """
        Initialize HTML formatter.
        
        Args:
            title: Table title
            css_class: CSS class for table
        """
        self.title = title
        self.css_class = css_class
    
    def format(self, results: List[Dict[str, Any]]) -> str:
        """Format results as HTML table."""
        if not results:
            return "<p>No results</p>"
        
        lines = []
        
        # Add basic styling
        lines.append("""<style>
.results-table { border-collapse: collapse; width: 100%; font-family: sans-serif; }
.results-table th, .results-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
.results-table th { background-color: #4CAF50; color: white; }
.results-table tr:nth-child(even) { background-color: #f2f2f2; }
.buy { color: green; font-weight: bold; }
.sell { color: red; font-weight: bold; }
</style>""")
        
        if self.title:
            lines.append(f"<h2>{self.title}</h2>")
        
        lines.append(f'<table class="{self.css_class}">')
        
        # Header
        headers = list(results[0].keys())
        lines.append("<tr>")
        for h in headers:
            lines.append(f"<th>{h.upper()}</th>")
        lines.append("</tr>")
        
        # Rows
        for result in results:
            lines.append("<tr>")
            for h in headers:
                value = result.get(h, "")
                css_class = ""
                if h == "signal":
                    css_class = ' class="buy"' if value in ("BUY", "STRONG_BUY") else ""
                    css_class = ' class="sell"' if value in ("SELL", "STRONG_SELL") else css_class
                lines.append(f"<td{css_class}>{value}</td>")
            lines.append("</tr>")
        
        lines.append("</table>")
        
        return "\n".join(lines)
    
    def get_content_type(self) -> str:
        return "text/html"


# Convenience functions
def format_csv(results: List[Dict[str, Any]]) -> str:
    """Format results as CSV."""
    return CSVFormatter().format(results)


def format_json(results: List[Dict[str, Any]], compact: bool = False) -> str:
    """Format results as JSON."""
    return JSONFormatter(compact=compact).format(results)


def format_table(results: List[Dict[str, Any]], title: str = None) -> str:
    """Format results as table."""
    return TableFormatter(title=title).format(results)


def format_html(results: List[Dict[str, Any]], title: str = None) -> str:
    """Format results as HTML."""
    return HTMLFormatter(title=title).format(results)
