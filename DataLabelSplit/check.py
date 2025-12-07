# debug_step1.py
import pandas as pd
import numpy as np

print("🔍 STEP 1: CHECK ORIGINAL DATA")
print("="*50)

# Load semua file
files = [
    r"C:\Users\LENOVO\OneDrive\Documents\SIC\Belajar\streetlight\MachineLearningSL\streetlight_data.csv",           # Data mentah
    r"C:\Users\LENOVO\OneDrive\Documents\SIC\Belajar\streetlight\MachineLearningSL\streetlight_data_cleaned.csv",   # Setelah cleaning
    r"C:\Users\LENOVO\OneDrive\Documents\SIC\Belajar\streetlight\MachineLearningSL\streetlight_labeled.csv",        # Setelah labeling
    r"C:\Users\LENOVO\OneDrive\Documents\SIC\Belajar\streetlight\MachineLearningSL\train_data.csv",                 # Train set
    r"C:\Users\LENOVO\OneDrive\Documents\SIC\Belajar\streetlight\MachineLearningSL\test_data.csv"                   # Test set
]

for file in files:
    try:
        df = pd.read_csv(file)
        print(f"\n📁 {file}:")
        print(f"   Rows: {len(df)}, Columns: {list(df.columns)}")
        print(f"   Sample:")
        print(df.head(2).to_string())
    except:
        print(f"\n❌ {file}: File not found")