"""
Email Module

Email notification functionality for Project Alpha.
Supports text, CSV, SVG, and image attachments.
"""

import datetime as dt
import json
import logging
import mimetypes
import os
import smtplib
from dataclasses import dataclass, field
from email.message import EmailMessage
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


logger = logging.getLogger(__name__)


@dataclass
class EmailConfig:
    """Email server configuration."""
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password: str
    sender_id: str
    recipient_list: List[str] = field(default_factory=list)
    
    @classmethod
    def from_json(cls, config_file: str) -> "EmailConfig":
        """Load configuration from JSON file."""
        with open(config_file, "r") as f:
            config = json.load(f)
        
        creds = config.get("email_credentials", {})
        return cls(
            smtp_host=creds.get("smtp_host", ""),
            smtp_port=creds.get("smtp_port", 587),
            smtp_user=creds.get("smtp_user", ""),
            smtp_password=creds.get("smtp_password", ""),
            sender_id=creds.get("sender_id", ""),
            recipient_list=config.get("recipient_list", []),
        )


class EmailServer:
    """
    Email server for sending notifications.
    
    Supports text, HTML, CSV, SVG, and image attachments.
    """
    
    def __init__(self, config: Union[str, EmailConfig]):
        """
        Initialize email server.
        
        Args:
            config: Path to config JSON or EmailConfig instance
        """
        if isinstance(config, str):
            self.config = EmailConfig.from_json(config)
        else:
            self.config = config
        
        self._server: Optional[smtplib.SMTP] = None
    
    def _connect(self) -> smtplib.SMTP:
        """Establish SMTP connection."""
        if self._server is None:
            self._server = smtplib.SMTP(
                self.config.smtp_host,
                self.config.smtp_port,
            )
            self._server.ehlo()
            self._server.starttls()
            self._server.login(
                self.config.smtp_user,
                self.config.smtp_password,
            )
        return self._server
    
    def disconnect(self):
        """Close SMTP connection."""
        if self._server:
            try:
                self._server.quit()
            except Exception:
                pass
            self._server = None
    
    def __del__(self):
        """Clean up on deletion."""
        self.disconnect()
    
    def send(
        self,
        subject: str,
        message: str,
        recipients: Optional[List[str]] = None,
        html: bool = True,
        mock: bool = True,
    ) -> bool:
        """
        Send a basic email.
        
        Args:
            subject: Email subject
            message: Email body
            recipients: Override recipient list
            html: Send as HTML
            mock: Log only, don't actually send
            
        Returns:
            True if sent successfully
        """
        recipients = recipients or self.config.recipient_list
        
        for recipient in recipients:
            try:
                msg = MIMEMultipart()
                msg["Subject"] = subject
                msg["From"] = self.config.sender_id
                msg["To"] = recipient
                
                content_type = "html" if html else "plain"
                msg.attach(MIMEText(message, content_type))
                
                if mock:
                    logger.info(f"Mock: Would send to {recipient}")
                else:
                    self._connect().sendmail(
                        msg["From"],
                        msg["To"],
                        msg.as_string().encode("utf-8"),
                    )
                    logger.info(f"Email sent to {recipient}")
                    
            except Exception as e:
                logger.error(f"Error sending email to {recipient}: {e}")
                return False
        
        return True
    
    def send_stock_report_email(
        self,
        subject: str,
        market: str,
        category: str,
        summary_data: List[Dict[str, Any]],
        charts: List[str],
        pdf_path: Optional[str] = None,
        recipients: Optional[List[str]] = None,
        mock: bool = True,
        extra_columns: Optional[List[str]] = None,
    ) -> bool:
        """
        Send a formatted stock report email with summary table and embedded charts.

        Args:
            subject: Email subject
            market: Market name
            category: Analysis category (e.g. Trend, Value)
            summary_data: List of dicts with keys 'symbol', 'price', 'change', 'sector'
            charts: List of paths to chart images (png/jpg) to embed
            recipients: Override recipient list
            mock: Log only, don't send

        Returns:
            True if sent successfully
        """
        recipients = recipients or self.config.recipient_list
        
        # Limit embedded charts to prevent huge emails
        MAX_EMBEDDED_CHARTS = 5
        embedded_charts = charts[:MAX_EMBEDDED_CHARTS]
        
        # Prepare dynamic columns
        extra_columns = extra_columns or []
        extra_headers = ""
        for col in extra_columns:
             extra_headers += f'<th style="padding: 10px; text-align: center;">{col}</th>'
        
        # Build HTML Body
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; color: #333;">
            <div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px;">
                <h2 style="color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
                    {category} Opportunities ({market.upper()})
                </h2>
                <p style="color: #7f8c8d;">Date: {dt.datetime.now().strftime("%Y-%m-%d")}</p>
                
                <h3>📊 Market Summary</h3>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px; background-color: white;">
                    <thead>
                        <tr style="background-color: #3498db; color: white;">
                            <th style="padding: 10px; text-align: left;">Symbol</th>
                            <th style="padding: 10px; text-align: right;">Price</th>
                            <th style="padding: 10px; text-align: right;">Change</th>
                            {extra_headers}
                            <th style="padding: 10px; text-align: left;">Sector</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for item in summary_data[:20]:
            change_color = "green" if item['change'] >= 0 else "red"
            change_str = f"{item['change']:.2f}%"
            if item['change'] > 0: change_str = "+" + change_str
            
            html_body += f"""
                        <tr style="border-bottom: 1px solid #bdc3c7;">
                            <td style="padding: 10px; font-weight: bold;">{item['symbol']}</td>
                            <td style="padding: 10px; text-align: right;">{item['price']}</td>
                            <td style="padding: 10px; text-align: right; color: {change_color};">{change_str}</td>
                            
                            <!-- Dynamic Cells -->
                            { "".join([f'<td style="padding: 10px; text-align: center;">{item.get(col, "")}</td>' for col in extra_columns]) }
                            
                            <td style="padding: 10px;">{item.get('sector', 'N/A')}</td>
                        </tr>
            """
            
        if len(summary_data) > 20:
             html_body += f"""
                        <tr>
                            <td colspan="4" style="padding: 10px; text-align: center; color: #7f8c8d; font-style: italic;">
                                ...and {len(summary_data) - 20} more items in the attached CSV...
                            </td>
                        </tr>
             """
            
        html_body += """
                    </tbody>
                </table>
        """
        
        if embedded_charts:
            html_body += "<h3>📈 Top Charts</h3>"
            for i, _ in enumerate(embedded_charts):
                html_body += f"""
                <div style="margin-bottom: 20px; border: 1px solid #ddd; padding: 10px; background-color: white;">
                    <img src="cid:chart_{i}" style="width: 100%; max-width: 800px; height: auto;">
                </div>
                """
        
        if len(charts) > MAX_EMBEDDED_CHARTS:
             html_body += f"<p><i>...and {len(charts) - MAX_EMBEDDED_CHARTS} more charts attached as files.</i></p>"

        html_body += """
            </div>
            <p style="font-size: 12px; color: #999; margin-top: 20px;">
                Generated by Project Alpha Automation
            </p>
        </body>
        </html>
        """

        # Build CSV for attachment
        csv_buffer = StringIO()
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(csv_buffer, index=False)
        
        for recipient in recipients:
            try:
                # Root is mixed: contains related (body+images) AND attachments (CSV)
                msg = MIMEMultipart("mixed")
                msg["Subject"] = subject
                msg["From"] = self.config.sender_id
                msg["To"] = recipient
                
                # Create the related part for HTML + Inline Images
                msg_related = MIMEMultipart("related")
                msg.attach(msg_related)
                
                # Alternative part for HTML text
                msg_alternative = MIMEMultipart("alternative")
                msg_related.attach(msg_alternative)
                msg_alternative.attach(MIMEText(html_body, "html"))
                
                # Embed images into the related part
                for i, chart_path in enumerate(embedded_charts):
                    if not os.path.exists(chart_path): continue
                    
                    with open(chart_path, "rb") as f:
                        img_data = f.read()
                        
                    # Try to guess MIME type or default to png
                    ctype, encoding = mimetypes.guess_type(chart_path)
                    if ctype is None or encoding is not None:
                        ctype = 'application/octet-stream'
                    
                    maintype, subtype = ctype.split('/', 1)
                    if maintype != 'image':
                        subtype = 'png'

                    img = MIMEImage(img_data, _subtype=subtype)
                    img.add_header("Content-ID", f"<chart_{i}>")
                    img.add_header("Content-Disposition", "inline", filename=os.path.basename(chart_path))
                    msg_related.attach(img)
                    
                # Attach CSV to the mixed root
                if summary_data:
                    timestamp = dt.datetime.now().strftime("%Y%m%d")
                    filename = f"stock_report_{market}_{category}_{timestamp}.csv"
                    attachment = MIMEApplication(
                        csv_buffer.getvalue().encode('utf-8'),
                        Name=filename,
                    )
                    attachment["Content-Disposition"] = f'attachment; filename="{filename}"'
                    msg.attach(attachment)

                # Attach PDF Report
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as f:
                        pdf_data = f.read()
                    
                    pdf_filename = os.path.basename(pdf_path)
                    pdf_attachment = MIMEApplication(pdf_data, _subtype="pdf")
                    pdf_attachment.add_header('Content-Disposition', 'attachment', filename=pdf_filename)
                    msg.attach(pdf_attachment)
                
                if mock:
                    logger.info(f"Mock: Would send report to {recipient} with {len(embedded_charts)} inline charts + CSV + PDF")
                else:
                    self._connect().sendmail(
                        msg["From"],
                        msg["To"],
                        msg.as_string().encode("utf-8"),
                    )
                    logger.info(f"Report sent to {recipient}")
                    
            except Exception as e:
                logger.error(f"Error sending report to {recipient}: {e}")
                return False
        
        return True
    
    def send_with_attachments(
        self,
        subject: str,
        message: str,
        attachments: List[str],
        recipients: Optional[List[str]] = None,
        mock: bool = True,
    ) -> bool:
        """
        Send email with multiple attachments.
        
        Args:
            subject: Email subject
            message: Email body
            attachments: List of file paths to attach
            recipients: Override recipient list
            mock: Log only, don't send
            
        Returns:
            True if sent successfully
        """
        recipients = recipients or self.config.recipient_list
        
        for recipient in recipients:
            try:
                msg = MIMEMultipart()
                msg["Subject"] = subject
                msg["From"] = self.config.sender_id
                msg["To"] = recipient
                msg.attach(MIMEText(message, "html"))
                
                # Attach each file
                for filepath in attachments:
                    if not os.path.exists(filepath):
                        continue
                    
                    mime_type, _ = mimetypes.guess_type(filepath)
                    filename = os.path.basename(filepath)
                    
                    with open(filepath, "rb") as f:
                        attachment = MIMEApplication(f.read(), Name=filename)
                        attachment["Content-Disposition"] = f'attachment; filename="{filename}"'
                        msg.attach(attachment)
                
                if mock:
                    logger.info(f"Mock: Would send {len(attachments)} attachments to {recipient}")
                else:
                    self._connect().sendmail(
                        msg["From"],
                        msg["To"],
                        msg.as_string().encode("utf-8"),
                    )
                    logger.info(f"Email with attachments sent to {recipient}")
                    
            except Exception as e:
                logger.error(f"Error sending email with attachments to {recipient}: {e}")
                return False
        
        return True


# Convenience functions
def send_analysis_email(
    market: str,
    analysis: str,
    output_dir: str,
    config_file: str = "email_config.json",
    mock: bool = False,
) -> bool:
    """
    Send analysis results via email.
    
    Args:
        market: Market name
        analysis: Analysis type
        output_dir: Directory with files to attach
        config_file: Email config JSON path
        mock: Log only, don't send
    """
    timestamp = dt.datetime.now().strftime("%d-%m-%y")
    
    server = EmailServer(config_file)
    subject = f"{analysis} for {market} on {timestamp}"
    message = f"<h2>{analysis} for {market}</h2><p>Date: {timestamp}</p>"
    
    # Find attachments
    attachments = []
    if os.path.isdir(output_dir):
        for f in os.listdir(output_dir):
            if f.endswith((".svg", ".png", ".jpg", ".csv")):
                attachments.append(os.path.join(output_dir, f))
    
    return server.send_with_attachments(subject, message, attachments, mock=mock)
