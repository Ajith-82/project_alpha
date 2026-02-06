"""
Output Package

Provides formatting, exporting, and visualization for screening results.
"""

from .formatters import (
    ResultFormatter,
    CSVFormatter,
    JSONFormatter,
    TableFormatter,
    HTMLFormatter,
    format_csv,
    format_json,
    format_table,
    format_html,
)
from .exporters import (
    Exporter,
    export_csv,
    export_json,
    export_html,
)
from .console import (
    console,
    COLORS,
    print_banner,
    print_section,
    print_success,
    print_error,
    print_warning,
    print_info,
    create_progress,
    create_download_progress,
    create_results_table,
    create_stock_table,
    print_summary_panel,
    print_config_panel,
    format_price,
    format_change,
    print_table,
)
from .email import (
    EmailConfig,
    EmailServer,
    send_analysis_email,
)
from .charts import (
    ChartConfig,
    ChartBuilder,
    create_stock_chart,
)


__all__ = [
    # Formatters
    "ResultFormatter",
    "CSVFormatter",
    "JSONFormatter",
    "TableFormatter",
    "HTMLFormatter",
    "format_csv",
    "format_json",
    "format_table",
    "format_html",
    # Exporters
    "Exporter",
    "export_csv",
    "export_json",
    "export_html",
    # Console
    "console",
    "COLORS",
    "print_banner",
    "print_section",
    "print_success",
    "print_error",
    "print_warning",
    "print_info",
    "create_progress",
    "create_download_progress",
    "create_results_table",
    "create_stock_table",
    "print_summary_panel",
    "print_config_panel",
    "format_price",
    "format_change",
    "print_table",
    # Email
    "EmailConfig",
    "EmailServer",
    "send_analysis_email",
    # Charts
    "ChartConfig",
    "ChartBuilder",
    "create_stock_chart",
]
