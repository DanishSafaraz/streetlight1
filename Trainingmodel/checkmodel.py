# check_models.py
import os
import pandas as pd
import joblib

print("📁 CHECKING MODELS_HARD FOLDER")
print("="*50)

models_hard_path = r"C:\Users\LENOVO\OneDrive\Documents\SIC\Belajar\streetlight\MachineLearningSL\Trainingmodel\models_hard"
if os.path.exists(models_hard_path):
    print(f"✅ Folder found: {models_hard_path}")
    
    # List semua file
    files = os.listdir(models_hard_path)
    print(f"\n📋 Files in folder:")
    for file in files:
        file_path = os.path.join(models_hard_path, file)
        if os.path.isdir(file_path):
            print(f"📁 {file}/")
            sub_files = os.listdir(file_path)
            for sub_file in sub_files:
                print(f"   └─ {sub_file}")
        else:
            size = os.path.getsize(file_path)
            print(f"📄 {file} ({size} bytes)")
    
    # Cek model files
    print("\n🤖 Checking model files...")
    model_files = [f for f in files if f.endswith('.pkl') and 'model' in f.lower()]
    
    if model_files:
        print(f"✅ Found {len(model_files)} model files:")
        for model_file in model_files:
            try:
                model = joblib.load(os.path.join(models_hard_path, model_file))
                print(f"   {model_file}: {type(model).__name__}")
            except:
                print(f"   {model_file}: Could not load")
    else:
        print("❌ No model files found!")
    
    # Cek comparison CSV
    csv_file = os.path.join(models_hard_path, "model_comparison_detailed.csv")
    if os.path.exists(csv_file):
        print(f"\n📊 Model comparison CSV found!")
        df = pd.read_csv(csv_file)
        print("   Metrics comparison:")
        print(df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].to_string(index=False))
    else:
        print("❌ No comparison CSV found")
    
    # Cek plots folder
    plots_path = os.path.join(models_hard_path, "plots")
    if os.path.exists(plots_path):
        plot_files = os.listdir(plots_path)
        print(f"\n🎨 Found {len(plot_files)} visualization files in plots/")
        for plot in plot_files[:5]:  # Show first 5
            print(f"   📈 {plot}")
        if len(plot_files) > 5:
            print(f"   ... and {len(plot_files) - 5} more")
    else:
        print("❌ No plots folder found")
        
else:
    print(f"❌ Folder not found: {models_hard_path}")