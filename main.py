# Main launcher to run the full pipeline

import subprocess
import sys

# Run the data pipeline
subprocess.run([sys.executable, "scripts/run_data_pipeline.py"])

# Run run the full analysis script
subprocess.run([sys.executable, "scripts/run_full_analysis.py"])