"""
Unit tests for the Output package.

Tests formatters, exporters, console utilities, and charts.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from classes.output.formatters import (
    CSVFormatter,
    JSONFormatter,
    TableFormatter,
    HTMLFormatter,
    format_csv,
    format_json,
)
from classes.output.exporters import Exporter


class TestCSVFormatter(unittest.TestCase):
    """Test cases for CSVFormatter."""
    
    def test_format_basic(self):
        """Test basic CSV formatting."""
        formatter = CSVFormatter()
        results = [
            {"ticker": "AAPL", "signal": "BUY", "confidence": 0.8},
            {"ticker": "MSFT", "signal": "SELL", "confidence": 0.6},
        ]
        
        output = formatter.format(results)
        
        self.assertIn("ticker", output)
        self.assertIn("AAPL", output)
        self.assertIn("BUY", output)
    
    def test_format_empty(self):
        """Test formatting empty list."""
        formatter = CSVFormatter()
        output = formatter.format([])
        self.assertEqual(output, "")
    
    def test_delimiter(self):
        """Test custom delimiter."""
        formatter = CSVFormatter(delimiter=";")
        results = [{"a": 1, "b": 2}]
        output = formatter.format(results)
        self.assertIn(";", output)
    
    def test_no_header(self):
        """Test without header."""
        formatter = CSVFormatter(include_header=False)
        results = [{"ticker": "AAPL"}]
        output = formatter.format(results)
        # Should only have one line (data, no header)
        lines = output.strip().split("\n")
        self.assertEqual(len(lines), 1)
    
    def test_content_type(self):
        """Test content type."""
        formatter = CSVFormatter()
        self.assertEqual(formatter.get_content_type(), "text/csv")


class TestJSONFormatter(unittest.TestCase):
    """Test cases for JSONFormatter."""
    
    def test_format_basic(self):
        """Test basic JSON formatting."""
        formatter = JSONFormatter()
        results = [{"ticker": "AAPL", "value": 100}]
        
        output = formatter.format(results)
        
        self.assertIn('"ticker"', output)
        self.assertIn('"AAPL"', output)
    
    def test_format_compact(self):
        """Test compact JSON."""
        formatter = JSONFormatter(compact=True)
        results = [{"a": 1}]
        output = formatter.format(results)
        # Compact should have no newlines
        self.assertNotIn("\n", output.strip())
    
    def test_content_type(self):
        """Test content type."""
        formatter = JSONFormatter()
        self.assertEqual(formatter.get_content_type(), "application/json")


class TestTableFormatter(unittest.TestCase):
    """Test cases for TableFormatter."""
    
    def test_format_basic(self):
        """Test basic table formatting."""
        formatter = TableFormatter()
        results = [{"ticker": "AAPL", "signal": "BUY"}]
        
        output = formatter.format(results)
        
        # Should contain column headers
        self.assertIn("TICKER", output.upper())
    
    def test_format_empty(self):
        """Test empty results."""
        formatter = TableFormatter()
        output = formatter.format([])
        self.assertEqual(output, "No results")
    
    def test_content_type(self):
        """Test content type."""
        formatter = TableFormatter()
        self.assertEqual(formatter.get_content_type(), "text/plain")


class TestHTMLFormatter(unittest.TestCase):
    """Test cases for HTMLFormatter."""
    
    def test_format_basic(self):
        """Test basic HTML formatting."""
        formatter = HTMLFormatter()
        results = [{"ticker": "AAPL", "signal": "BUY"}]
        
        output = formatter.format(results)
        
        self.assertIn("<table", output)
        self.assertIn("AAPL", output)
        self.assertIn("</table>", output)
    
    def test_format_with_title(self):
        """Test HTML with title."""
        formatter = HTMLFormatter(title="Test Results")
        results = [{"a": 1}]
        output = formatter.format(results)
        self.assertIn("Test Results", output)
    
    def test_signal_coloring(self):
        """Test signal color classes."""
        formatter = HTMLFormatter()
        results = [{"signal": "BUY"}]
        output = formatter.format(results)
        self.assertIn('class="buy"', output)
    
    def test_content_type(self):
        """Test content type."""
        formatter = HTMLFormatter()
        self.assertEqual(formatter.get_content_type(), "text/html")


class TestExporter(unittest.TestCase):
    """Test cases for Exporter."""
    
    def setUp(self):
        """Create temp directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.exporter = Exporter(output_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up temp files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_to_csv(self):
        """Test CSV export."""
        results = [{"ticker": "AAPL", "signal": "BUY"}]
        path = self.exporter.to_csv(results, "test.csv")
        
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            content = f.read()
        self.assertIn("AAPL", content)
    
    def test_to_json(self):
        """Test JSON export."""
        results = [{"ticker": "AAPL", "value": 100}]
        path = self.exporter.to_json(results, "test.json")
        
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            content = f.read()
        self.assertIn("AAPL", content)
    
    def test_to_html(self):
        """Test HTML export."""
        results = [{"ticker": "AAPL"}]
        path = self.exporter.to_html(results, "test.html", title="Test")
        
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            content = f.read()
        self.assertIn("<!DOCTYPE html>", content)
        self.assertIn("AAPL", content)
    
    def test_export_auto_detect(self):
        """Test auto format detection."""
        results = [{"a": 1}]
        
        # CSV
        path = self.exporter.export(results, "test.csv")
        self.assertTrue(path.endswith(".csv"))
        
        # JSON
        path = self.exporter.export(results, "test.json")
        self.assertTrue(path.endswith(".json"))


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def test_format_csv(self):
        """Test format_csv function."""
        output = format_csv([{"a": 1, "b": 2}])
        self.assertIn("a", output)
    
    def test_format_json(self):
        """Test format_json function."""
        output = format_json([{"a": 1}])
        self.assertIn('"a"', output)


class TestConsoleUtilities(unittest.TestCase):
    """Test console utility functions."""
    
    def test_format_price(self):
        """Test price formatting."""
        from classes.output.console import format_price
        result = format_price(1234.56)
        self.assertIn("1,234.56", result)
    
    def test_format_change_positive(self):
        """Test positive change formatting."""
        from classes.output.console import format_change
        result = format_change(5.5)
        self.assertIn("5.50", result)
    
    def test_format_change_negative(self):
        """Test negative change formatting."""
        from classes.output.console import format_change
        result = format_change(-3.2)
        self.assertIn("-3.20", result)


class TestChartBuilder(unittest.TestCase):
    """Test ChartBuilder class."""
    
    def setUp(self):
        """Create sample data."""
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range("2024-01-01", periods=30, freq="D")
        self.data = pd.DataFrame({
            "Open": np.random.uniform(100, 110, 30),
            "High": np.random.uniform(105, 115, 30),
            "Low": np.random.uniform(95, 105, 30),
            "Close": np.random.uniform(100, 110, 30),
            "Volume": np.random.uniform(1000000, 2000000, 30),
        }, index=dates)
    
    def test_create_chart(self):
        """Test chart creation."""
        from classes.output.charts import ChartBuilder, ChartConfig
        
        config = ChartConfig(show_volume=True)
        builder = ChartBuilder(config)
        builder.create("AAPL", self.data)
        
        fig = builder.get_figure()
        # Should have created a figure
        self.assertIsNotNone(fig)
    
    def test_save_chart(self):
        """Test chart saving."""
        from classes.output.charts import ChartBuilder
        
        builder = ChartBuilder()
        builder.create("AAPL", self.data)
        
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        
        try:
            result = builder.save(path)
            self.assertTrue(os.path.exists(result))
        finally:
            if os.path.exists(path):
                os.unlink(path)


if __name__ == "__main__":
    unittest.main()
