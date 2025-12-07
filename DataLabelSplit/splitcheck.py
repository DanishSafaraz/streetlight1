# debug_step2.py
import pandas as pd
import numpy as np

print("🔍 STEP 2: CHECK SPLIT PROBLEM")
print("="*50)

# Load train and test
train = pd.read_csv(r"C:\Users\LENOVO\OneDrive\Documents\SIC\Belajar\streetlight\MachineLearningSL\train_data.csv")
test = pd.read_csv(r"C:\Users\LENOVO\OneDrive\Documents\SIC\Belajar\streetlight\MachineLearningSL\test_data.csv")

print(f"Train size: {len(train)}")
print(f"Test size:  {len(test)}")

# Cek duplikat antara train dan test
print("\n🔎 Checking duplicates between train and test...")
train_test_combined = pd.concat([train, test])
duplicates = train_test_combined[train_test_combined.duplicated()]
print(f"Duplicates found: {len(duplicates)}")

if len(duplicates) > 0:
    print("\n❌ PROBLEM: Ada data yang sama di train dan test!")
    print("Sample duplicates:")
    print(duplicates.head())
    
    # Tampilkan persentase
    total_duplicates = len(duplicates)
    total_data = len(train) + len(test)
    print(f"\n📊 {total_duplicates}/{total_data} ({total_duplicates/total_data*100:.1f}%) data duplikat!")
    
    # Solusi: Hapus duplikat dari test
    print("\n💡 SOLUTION: Remove duplicates from test set")
    test_clean = test.drop_duplicates()
    # Cek lagi apakah masih ada overlap dengan train
    test_only = test_clean[~test_clean.isin(train)].dropna()
    print(f"Test after cleaning: {len(test_only)} rows")
    
    # Save clean test
    test_only.to_csv("test_data_clean.csv", index=False)
    print("💾 Saved: test_data_clean.csv")
else:
    print("✅ No duplicates between train and test")