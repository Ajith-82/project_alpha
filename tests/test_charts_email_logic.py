
import os
import pandas as pd
from classes.output.charts import create_batch_charts
from classes.output.email import EmailConfig, EmailServer

# Mock data
symbols = ["TEST1", "TEST2"]
price_data = {
    "TEST1": pd.DataFrame({
        "Open": [100, 101],
        "High": [102, 103],
        "Low": [99, 100],
        "Close": [101, 102],
        "Volume": [1000, 2000]
    }, index=pd.to_datetime(["2023-01-01", "2023-01-02"])),
    "TEST2": pd.DataFrame({
        "Open": [50, 51],
        "High": [52, 53],
        "Low": [49, 50],
        "Close": [51, 52],
        "Volume": [500, 600]
    }, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
}
data = {"price_data": price_data, "sectors": {"TEST1": "Tech", "TEST2": "Finance"}}
output_dir = "tests/test_charts_output"

# Ensure config exists for the test to attempt sending
if not os.path.exists("email_config.json"):
    print("WARNING: email_config.json not found, test might skip email sending block")

print("Running create_batch_charts...")
try:
    create_batch_charts(
        screener_name="Test Screener",
        market="india",
        symbols=symbols,
        data=data,
        output_dir=output_dir,
        batch_size=10,
        send_email_flag=True
    )
    print("create_batch_charts completed successfully!")
except Exception as e:
    print(f"create_batch_charts FAILED: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
import shutil
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
