
import os
import pandas as pd
import shutil
from classes.output.charts import create_batch_charts
from classes.output.email import EmailConfig, EmailServer

# Mock data - Create 60 symbols to trigger splitting (Threshold is 50)
symbols = [f"TEST{i}" for i in range(1, 61)]
price_data = {}
sectors = {}
metadata = {}

# Create dummy data for all 60
for sym in symbols:
    price_data[sym] = pd.DataFrame({
        "Open": [100, 101],
        "High": [102, 103],
        "Low": [99, 100],
        "Close": [101, 102],
        "Volume": [1000, 2000]
    }, index=pd.to_datetime(["2023-01-01", "2023-01-02"]))
    sectors[sym] = "Tech"
    metadata[sym] = {"Growth": "0.5%", "Vol": "0.05", "Score": "9.5", "Signal": "Buy"}

data = {"price_data": price_data, "sectors": sectors}
output_dir = "tests/test_charts_output"

# Ensure config exists for the test to attempt sending
if not os.path.exists("email_config.json"):
    print("WARNING: email_config.json not found, test might skip email sending block")
else:
    print("email_config.json found, will simulate sending.")

# Clean previous
if os.path.exists(output_dir + "_batch_0"):
    shutil.rmtree(output_dir + "_batch_0")

# MOCK _create_full_chart to speed up test
from unittest.mock import patch

def mock_create_chart(ticker, data, output_path, market):
    # Valid 1x1 transparent PNG
    import base64
    png_data = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=")
    with open(output_path, "wb") as f:
        f.write(png_data)

print("Running create_batch_charts with 60 symbols (MOCKED CHARTS)...")
try:
    with patch("classes.output.charts._create_full_chart", side_effect=mock_create_chart):
        create_batch_charts(
            screener_name="Test Screener",
            market="india",
            symbols=symbols,
            data=data,
            output_dir=output_dir,
            batch_size=100, # Batch size for chart generation, not email
            send_email_flag=True,
            analysis_metadata=metadata
        )
    print("create_batch_charts completed successfully!")
    
    # Verify PDF creation (Part 1 and Part 2)
    part1 = os.path.join(output_dir + "_batch_0", "Test Screener_Report_Part1.pdf")
    part2 = os.path.join(output_dir + "_batch_0", "Test Screener_Report_Part2.pdf")
    
    if os.path.exists(part1) and os.path.exists(part2):
        print(f"SUCCESS: Both PDF Parts generated:\n  {part1}\n  {part2}")
    else:
        print(f"FAILURE: PDFs missing.\n  Part1 exists: {os.path.exists(part1)}\n  Part2 exists: {os.path.exists(part2)}")
        # Check if single report exists (failure case)
        single = os.path.join(output_dir + "_batch_0", "Test Screener_Report.pdf")
        if os.path.exists(single):
            print(f"  Single report found instead: {single}")
        
except Exception as e:
    print(f"create_batch_charts FAILED: {e}")
    import traceback
    traceback.print_exc()

# Cleanup
if os.path.exists(output_dir + "_batch_0"):
    shutil.rmtree(output_dir + "_batch_0")
