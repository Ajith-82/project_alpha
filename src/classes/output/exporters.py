"""
Result Exporters

Provides file export functionality for screening results.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .formatters import CSVFormatter, JSONFormatter, HTMLFormatter, TableFormatter


logger = logging.getLogger(__name__)


class Exporter:
    """
    Export screening results to various file formats.
    
    Supports CSV, JSON, HTML, and plain text exports.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize exporter.
        
        Args:
            output_dir: Default output directory
        """
        self.output_dir = output_dir or os.getcwd()
    
    def _ensure_dir(self, path: str) -> None:
        """Ensure parent directory exists."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, filename: str) -> str:
        """Get full path for filename."""
        if os.path.isabs(filename):
            return filename
        return os.path.join(self.output_dir, filename)
    
    def to_csv(
        self,
        results: List[Dict[str, Any]],
        filename: str = "results.csv",
        delimiter: str = ",",
    ) -> str:
        """
        Export results to CSV file.
        
        Args:
            results: List of result dictionaries
            filename: Output filename
            delimiter: Field delimiter
            
        Returns:
            Path to saved file
        """
        path = self._get_path(filename)
        self._ensure_dir(path)
        
        formatter = CSVFormatter(delimiter=delimiter)
        content = formatter.format(results)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Exported {len(results)} results to {path}")
        return path
    
    def to_json(
        self,
        results: List[Dict[str, Any]],
        filename: str = "results.json",
        indent: int = 2,
    ) -> str:
        """
        Export results to JSON file.
        
        Args:
            results: List of result dictionaries
            filename: Output filename
            indent: JSON indentation
            
        Returns:
            Path to saved file
        """
        path = self._get_path(filename)
        self._ensure_dir(path)
        
        formatter = JSONFormatter(indent=indent)
        content = formatter.format(results)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Exported {len(results)} results to {path}")
        return path
    
    def to_html(
        self,
        results: List[Dict[str, Any]],
        filename: str = "results.html",
        title: Optional[str] = "Screening Results",
    ) -> str:
        """
        Export results to HTML file.
        
        Args:
            results: List of result dictionaries
            filename: Output filename
            title: Page title
            
        Returns:
            Path to saved file
        """
        path = self._get_path(filename)
        self._ensure_dir(path)
        
        formatter = HTMLFormatter(title=title)
        content = formatter.format(results)
        
        # Wrap in full HTML document
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title or 'Results'}</title>
</head>
<body>
{content}
</body>
</html>"""
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        
        logger.info(f"Exported {len(results)} results to {path}")
        return path
    
    def to_text(
        self,
        results: List[Dict[str, Any]],
        filename: str = "results.txt",
        title: Optional[str] = None,
    ) -> str:
        """
        Export results to plain text file.
        
        Args:
            results: List of result dictionaries
            filename: Output filename
            title: Optional title
            
        Returns:
            Path to saved file
        """
        path = self._get_path(filename)
        self._ensure_dir(path)
        
        formatter = TableFormatter(title=title)
        content = formatter.format(results)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Exported {len(results)} results to {path}")
        return path
    
    def export(
        self,
        results: List[Dict[str, Any]],
        filename: str,
        format: Optional[str] = None,
    ) -> str:
        """
        Export results with auto-detected format.
        
        Args:
            results: List of result dictionaries
            filename: Output filename (format detected from extension)
            format: Override format (csv, json, html, txt)
            
        Returns:
            Path to saved file
        """
        # Detect format from extension if not specified
        if format is None:
            ext = Path(filename).suffix.lower()
            format_map = {
                ".csv": "csv",
                ".json": "json",
                ".html": "html",
                ".htm": "html",
                ".txt": "txt",
            }
            format = format_map.get(ext, "csv")
        
        export_methods = {
            "csv": self.to_csv,
            "json": self.to_json,
            "html": self.to_html,
            "txt": self.to_text,
        }
        
        method = export_methods.get(format.lower(), self.to_csv)
        return method(results, filename)


# Convenience functions
def export_csv(results: List[Dict[str, Any]], path: str) -> str:
    """Export results to CSV."""
    return Exporter().to_csv(results, path)


def export_json(results: List[Dict[str, Any]], path: str) -> str:
    """Export results to JSON."""
    return Exporter().to_json(results, path)


def export_html(results: List[Dict[str, Any]], path: str, title: str = None) -> str:
    """Export results to HTML."""
    return Exporter().to_html(results, path, title)
