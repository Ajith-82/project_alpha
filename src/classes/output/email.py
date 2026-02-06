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
    
    def send_with_csv(
        self,
        subject: str,
        message: str,
        df: pd.DataFrame,
        filename: str = "data.csv",
        recipients: Optional[List[str]] = None,
        mock: bool = True,
    ) -> bool:
        """
        Send email with CSV attachment.
        
        Args:
            subject: Email subject
            message: Email body
            df: DataFrame to attach as CSV
            filename: Attachment filename
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
                
                # Attach CSV
                csv_buffer = StringIO()
                df.to_csv(csv_buffer, index=False)
                attachment = MIMEApplication(
                    csv_buffer.getvalue(),
                    Name=filename,
                )
                attachment["Content-Disposition"] = f'attachment; filename="{filename}"'
                msg.attach(attachment)
                
                if mock:
                    logger.info(f"Mock: Would send CSV to {recipient}")
                else:
                    self._connect().sendmail(
                        msg["From"],
                        msg["To"],
                        msg.as_string().encode("utf-8"),
                    )
                    logger.info(f"Email with CSV sent to {recipient}")
                    
            except Exception as e:
                logger.error(f"Error sending email with CSV to {recipient}: {e}")
                return False
        
        return True
    
    def send_with_image(
        self,
        subject: str,
        message: str,
        image_path: str,
        recipients: Optional[List[str]] = None,
        mock: bool = True,
    ) -> bool:
        """
        Send email with image attachment.
        
        Args:
            subject: Email subject
            message: Email body
            image_path: Path to image file
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
                
                # Attach image
                with open(image_path, "rb") as f:
                    image = MIMEImage(
                        f.read(),
                        name=os.path.basename(image_path),
                    )
                    msg.attach(image)
                
                if mock:
                    logger.info(f"Mock: Would send image to {recipient}")
                else:
                    self._connect().sendmail(
                        msg["From"],
                        msg["To"],
                        msg.as_string().encode("utf-8"),
                    )
                    logger.info(f"Email with image sent to {recipient}")
                    
            except Exception as e:
                logger.error(f"Error sending email with image to {recipient}: {e}")
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
