
import os
import io
import pandas as pd
from classes.output.email import EmailConfig, EmailServer

# Create dummy chart
chart_path = "debug_chart.png"
with open(chart_path, "wb") as f:
    f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")

# Mock data
summary_data = [{"symbol": "TEST", "price": "100", "change": 5.0, "sector": "Tech"}]

# Mock Config
config = EmailConfig(
    smtp_host="localhost",
    smtp_port=25,
    smtp_user="user",
    smtp_password="pwd",
    sender_id="sender@example.com",
    recipient_list=["recipient@example.com"]
)

server = EmailServer(config)

# We want to inspect the generated message, but send_stock_report_email doesn't return it.
# We will duplicate the logic slightly or use a monkeypatch if needed, 
# but easiest is to copy the relevant block here to inspect 'msg' string.

import mimetypes
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from io import StringIO
import datetime as dt

# --- REPLICATING LOGIC FROM email.py for inspection ---
subject = "Debug Email"
market = "us"
category = "Trend"
extra_text = "Debug"
embedded_charts = [chart_path]
recipients = ["test@test.com"]
html_body = "<html><body><img src='cid:chart_0'></body></html>"

csv_buffer = StringIO()
df = pd.DataFrame(summary_data)
df.to_csv(csv_buffer, index=False)

msg = MIMEMultipart("mixed")
msg["Subject"] = subject
msg["From"] = config.sender_id
msg["To"] = recipients[0]

# Related
msg_related = MIMEMultipart("related")
msg.attach(msg_related)

# Alternative
msg_alternative = MIMEMultipart("alternative")
msg_related.attach(msg_alternative)

# Plain text fallback (MISSING IN PROD?)
msg_alternative.attach(MIMEText("Please view in HTML", "plain"))

msg_alternative.attach(MIMEText(html_body, "html"))

# Embed image
for i, path in enumerate(embedded_charts):
    with open(path, "rb") as f:
        img_data = f.read()
    
    img = MIMEImage(img_data, _subtype="png")
    img.add_header("Content-ID", f"<chart_{i}>")
    img.add_header("Content-Disposition", "inline", filename=os.path.basename(path))
    msg_related.attach(img)

# Attachment
attachment = MIMEApplication(csv_buffer.getvalue().encode('utf-8'), Name="report.csv")
attachment["Content-Disposition"] = 'attachment; filename="report.csv"'
msg.attach(attachment)

print("--- MIME STRUCTURE ---")
print(msg.as_string())

# Cleanup
if os.path.exists(chart_path):
    os.remove(chart_path)
