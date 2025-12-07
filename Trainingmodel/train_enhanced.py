# train_enhanced.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_with_enhanced_features():
    """
    Train model dengan enhanced features
    """
    print("="*60)
    print("TRAINING WITH ENHANCED FEATURES")
    print("="*60)
    
    # Load enhanced data
    data_file = r"C:\Users\LENOVO\OneDrive\Documents\SIC\Belajar\streetlight\MachineLearningSL\Trainingmodel\streetlight_enhanced.csv"
    if not os.path.exists(data_file):
        print(f"❌ File tidak ditemukan: {data_file}")
        print("   Jalankan enhance_features.py terlebih dahulu!")
        return
    
    df = pd.read_csv(data_file)
    print(f"📥 Loaded {len(df)} rows, {len(df.columns)} features")
    
    # Features untuk training
    # Pilih features yang numeric
    exclude_cols = ['timestamp', 'class', 'light_category']  # Kolom non-numeric
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"\n🔧 Selected {len(feature_cols)} features:")
    for i, feat in enumerate(feature_cols, 1):
        print(f"   {i:2d}. {feat}")
    
    X = df[feature_cols]
    y = df['class']  # Target
    
    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"\n🎯 Target classes: {le.classes_}")
    print(f"   Class distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\n✂️ Data split:")
    print(f"   Train: {X_train.shape}")
    print(f"   Test:  {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    results = []
    
    print("\n🤖 Training models...")
    for name, model in models.items():
        print(f"\n   {name}:")
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        
        results.append({
            'Model': name,
            'Accuracy': accuracy
        })
        
        # Save model
        model_file = f"models/{name.lower().replace(' ', '_')}_enhanced.pkl"
        joblib.dump(model, model_file)
        print(f"   💾 Saved: {model_file}")
    
    # Results summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
    print(results_df.to_string(index=False))
    
    # Realistic expectation check
    best_acc = results_df.iloc[0]['Accuracy']
    if best_acc > 0.95:
        print(f"\n⚠ Still too high: {best_acc:.1%}")
        print("   Consider making task harder")
    elif best_acc < 0.7:
        print(f"\n⚠ Might be too low: {best_acc:.1%}")
        print("   Check feature quality")
    else:
        print(f"\n✅ Good realistic accuracy: {best_acc:.1%}")
        print("   Expected range: 70-90%")
    
    return results_df

if __name__ == "__main__":
    # Buat folder models jika belum ada
    os.makedirs('models', exist_ok=True)
    
    train_with_enhanced_features()