import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.message import EmailMessage
import mimetypes
from io import StringIO
import logging
import json
import datetime as dt

from pandas import read_csv

class EmailServer:

    def __init__(self, config_file):
        # Read configuration from JSON file
        with open(config_file, 'r') as json_file:
            config = json.load(json_file)

        # Email credentials
        email_credentials = config.get('email_credentials', {})
        self.smtp_host = email_credentials.get('smtp_host')
        self.smtp_port = email_credentials.get('smtp_port')
        self.smtp_user = email_credentials.get('smtp_user')
        self.smtp_password = email_credentials.get('smtp_password')
        self.sender_id = email_credentials.get('sender_id')

        # Recipient list
        self.recipient_list = config.get('recipient_list', [])

        # Set up SMTP connection
        self.server = smtplib.SMTP(self.smtp_host, self.smtp_port)
        self.server.ehlo()
        self.server.starttls()
        self.server.login(self.smtp_user, self.smtp_password)

    def __del__(self):
        # Close SMTP connection when the object is deleted
        if hasattr(self, 'server'):
            self.server.quit()

    def send(self, subject, message, recipient_list=None, mock=True, svg_folder=None):
        if recipient_list is None:
            recipient_list = self.recipient_list

        for recipient in recipient_list:
            try:
                # Email content setup
                msg = MIMEMultipart()
                msg['Subject'] = subject
                msg['From'] = self.sender_id
                msg['To'] = recipient
                msg.add_header('Content-Type', 'text/html')
                msg.attach(MIMEText(message, 'html'))

                # Attach SVG files from the specified folder
                if svg_folder:
                    svg_files = [f for f in os.listdir(svg_folder) if f.lower().endswith('.svg')]
                    for svg_file in svg_files:
                        svg_path = os.path.join(svg_folder, svg_file)
                        with open(svg_path, 'r') as svg_file_content:
                            svg_content = svg_file_content.read()
                        svg_attachment = MIMEText(svg_content, 'html')
                        svg_attachment.add_header('Content-Disposition', 'inline', filename=svg_file)
                        msg.attach(svg_attachment)

                if mock:
                    logging.info('Mock: Sent to {}'.format(recipient))
                else:
                    self.server.sendmail(msg['From'], msg['To'], msg.as_string().encode('utf-8'))
                    logging.info(f"Email with attachment sent successfully to {recipient}")
            except Exception as e:
                logging.error(f"Error sending email to {recipient}: {e}")

    def send_csv_attachment(self, subject, message, filename, df, recipient_list=None, mock=True):
        if recipient_list is None:
            recipient_list = self.recipient_list

        for recipient in recipient_list:
            try:
                # Email content setup
                msg = MIMEMultipart()
                msg['Subject'] = subject
                msg['From'] = self.sender_id
                msg['To'] = recipient
                msg.add_header('Content-Type', 'text/html')
                msg.attach(MIMEText(message, 'html'))

                # Attach CSV file
                with StringIO() as text_stream:
                    df.to_csv(text_stream, index=False)
                    msg.attach(MIMEApplication(text_stream.getvalue(), Name=filename))

                if mock:
                    logging.info('Mock: Sent to {}'.format(recipient))
                else:
                    self.server.sendmail(msg['From'], msg['To'], msg.as_string().encode('utf-8'))
                    logging.info(f"Email with attachment sent successfully to {recipient}")
            except Exception as e:
                logging.error(f"Error sending email with attachment to {recipient}: {e}")

    def send_svg_attachment(self, subject, message, svg_folder, recipient_list=None, mock=True):
        """Send email with SVG files attached and embedded in the message."""
        if recipient_list is None:
            recipient_list = self.recipient_list
        for recipient in recipient_list:
            try:
                # email content set up
                msg = EmailMessage()
                msg['Subject'] = subject
                msg['From'] = self.sender_id
                msg['To'] = recipient
                msg.add_header('Content-Type', 'text/html')
                msg.set_content(message, subtype='html')

                # Attach and embed SVG files
                for svg_file in os.listdir(svg_folder):
                    if svg_file.endswith(".svg"):
                        svg_path = os.path.join(svg_folder, svg_file)
                        with open(svg_path, 'rb') as svg_file:
                            svg_content = svg_file.read()
                            mime_type, _ = mimetypes.guess_type(svg_path)
                            msg.add_attachment(svg_content, maintype='image', subtype=mime_type, filename=os.path.basename(svg_path))

                if mock:
                    logging.info(f'Mock: Sent to {recipient}')
                else:
                    self.server.sendmail(msg['From'], msg['To'], msg.as_bytes())
                    logging.info(f"Email with attachment sent successfully to {recipient}")

            except Exception as e:
                logging.error(f"Error sending email with CSV and images to {recipient}: {e}")


    def send_image_attachment(
        self,
        subject: str,
        message: str,
        image_file_path: str = None,
        recipients: list = None,
        mock: bool = True
    ) -> None:
        """
        Sends an email with an image attachment.

        Args:
            subject (str): The subject of the email.
            message (str): The message body of the email.
            image_file_path (Optional[str]): The path to the image file.
            recipients (Optional[List[str]]): The list of recipient email addresses.
            mock (bool): Whether to send a mock email or not.

        Returns:
            None

        Raises:
            Exception: If an error occurs while sending the email.
        """
        recipients = recipients or self.recipient_list

        for recipient in recipients:
            try:
                email = MIMEMultipart()
                email['Subject'] = subject
                email['From'] = self.sender_id
                email['To'] = recipient
                email.add_header('Content-Type', 'text/html')
                email.attach(MIMEText(message, 'html'))

                if image_file_path:
                    with open(image_file_path, 'rb') as image_file:
                        image = MIMEImage(image_file.read(), name=os.path.basename(image_file_path))
                        email.attach(image)

                if mock:
                    logging.info(f'Mock: Sent to {recipient}')
                else:
                    self.server.sendmail(email['From'], email['To'], email.as_string().encode('utf-8'))
                    logging.info(f"Email with attachment sent successfully to {recipient}")
            except Exception as e:
                logging.error(f"Error sending email with image attachment to {recipient}: {e}")

# Customized email functions
def send_email(market: str, analysis: str, out_dir: str):
    # Send email with reports
    timestamp = dt.datetime.now().strftime("%d-%m-%y")
    email_server = EmailServer("email_config.json")
    email_subject = f"{analysis} for {market} on {timestamp}"
    email_message = f"{analysis} for {market} on {timestamp}"
    email_server.send_svg_attachment(
        email_subject,
        email_message,
        svg_folder=out_dir,
        mock=False,
    )

def send_email_volatile(market: str, out_dir: str):
    # Check and select volatile output files
    files = os.listdir(out_dir)
    # Select all png files
    png_files = [f for f in files if f.endswith('.png')]
    # Select each file and send an email
    for file in png_files:
        # Set analysis name from file name
        analysis = file.split('.')[0]
        #print("Sending email for Volatile", analysis)
        # Send email with volatile reports
        timestamp = dt.datetime.now().strftime("%d-%m-%y")
        email_server = EmailServer("email_config.json")
        email_subject = f"{analysis} for {market} on {timestamp}"
        email_message = f"Today's {analysis} for {market}"
        file_path=os.path.join(out_dir, file)
        email_server.send_image_attachment(
            email_subject,
            email_message,
            image_file_path=file_path,
            mock=False,
        )