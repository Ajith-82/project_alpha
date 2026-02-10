import os
import matplotlib.pyplot as plt
from classes.output.email import EmailConfig, EmailServer

# Create a dummy chart
chart_path = "test_chart.png"
plt.figure()
plt.plot([1, 2, 3], [1, 2, 3])
plt.title("Test Chart")
plt.savefig(chart_path)
plt.close()

# Config
config = EmailConfig.from_json("email_config.json")
server = EmailServer(config)

# Dummy summary data
summary_data = [
    {"symbol": "TEST", "price": "100.00", "change": 5.0, "sector": "Technology"},
    {"symbol": "FAIL", "price": "50.00", "change": -2.0, "sector": "Energy"}
]

# Send email
success = server.send_stock_report_email(
    subject="Test Inline Charts",
    market="test",
    category="Test Category",
    summary_data=summary_data,
    charts=[chart_path],
    mock=False
)

if success:
    print("Email sent successfully!")
else:
    print("Failed to send email.")

# Cleanup
if os.path.exists(chart_path):
    os.remove(chart_path)
