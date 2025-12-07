# train_hard.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

# Setup style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def create_hard_task_dataset():
    """
    Buat dataset dengan task prediction yang lebih sulit
    """
    print("="*60)
    print("CREATING HARD PREDICTION TASK DATASET")
    print("="*60)
    
    # Load enhanced data
    data_file = r"C:\Users\LENOVO\OneDrive\Documents\SIC\Belajar\streetlight\MachineLearningSL\Trainingmodel\streetlight_enhanced.csv"
    if not os.path.exists(data_file):
        print(f"❌ File tidak ditemukan: {data_file}")
        print("   Jalankan enhance_features.py terlebih dahulu!")
        return None
    
    df = pd.read_csv(data_file)
    print(f"📥 Loaded {len(df)} samples")
    
    # TASK 1: Predict NEXT TIME PERIOD'S STATE (lebih sulit dari klasifikasi biasa)
    print("\n🎯 Creating hard prediction task: NEXT PERIOD STATE")
    
    # Shift data untuk prediksi periode berikutnya
    shift_period = 3  # Predict 3 periods ahead (30 menit jika interval 10 menit)
    
    # Features untuk prediksi (current state)
    feature_cols = ['light_intensity', 'voltage', 'hour', 'hour_sin', 'hour_cos',
                   'intensity_rolling_mean', 'intensity_rolling_std', 
                   'lamp_on_duration', 'energy_estimate']
    
    # Target: state di periode berikutnya (shifted)
    df['next_period_class'] = df['class'].shift(-shift_period)
    
    # Drop rows dengan NaN (akhir dataset)
    df_hard = df.dropna(subset=['next_period_class']).copy()
    
    X = df_hard[feature_cols]
    y = df_hard['next_period_class']
    
    print(f"   Task: Predict class {shift_period} periods ahead")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Samples after shift: {len(df_hard)}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")
    
    # Save hard task dataset
    hard_data = pd.concat([X, y], axis=1)
    hard_data.to_csv("streetlight_hard_task.csv", index=False)
    print(f"\n💾 Hard task dataset saved: streetlight_hard_task.csv")
    
    return X, y, feature_cols

def train_and_evaluate_hard_task():
    """
    Train dan evaluate 3 model pada hard prediction task
    """
    print("\n" + "="*70)
    print("HARD TASK MODEL TRAINING & EVALUATION")
    print("="*70)
    
    # Buat atau load hard task dataset
    hard_file = "streetlight_hard_task.csv"
    if os.path.exists(hard_file):
        print(f"📥 Loading hard task dataset: {hard_file}")
        hard_data = pd.read_csv(hard_file)
        
        # Pisahkan features dan target
        target_col = 'next_period_class' if 'next_period_class' in hard_data.columns else 'class'
        X = hard_data.drop(columns=[target_col])
        y = hard_data[target_col]
        feature_cols = X.columns.tolist()
    else:
        print("⚠ Hard task dataset not found, creating new one...")
        X, y, feature_cols = create_hard_task_dataset()
        if X is None:
            return
    
    print(f"\n📊 Dataset info:")
    print(f"   Samples: {X.shape[0]}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Target classes: {y.unique()}")
    print(f"   Class distribution:")
    print(y.value_counts())
    
    # 1. SPLIT DATA
    print("\n✂️ 1. Splitting data (70% train, 15% validation, 15% test)...")
    
    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Second split: 50/50 dari temp untuk validation dan test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"   Train:      {X_train.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")
    print(f"   Test:       {X_test.shape[0]} samples")
    
    # 2. PREPROCESSING
    print("\n🔧 2. Preprocessing data...")
    
    # Encode target
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_val_encoded = le.transform(y_val)
    y_test_encoded = le.transform(y_test)
    
    joblib.dump(le, 'models_hard/target_encoder.pkl')
    print(f"   Target encoder saved")
    print(f"   Classes: {le.classes_}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    joblib.dump(scaler, 'models_hard/feature_scaler.pkl')
    print(f"   Feature scaler saved")
    
    # 3. TRAIN MODELS
    print("\n🤖 3. Training 3 models...")
    
    models = {
        'Decision Tree': DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            metric='minkowski'
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=2000,
            C=0.1,
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs'
        )
    }
    
    # Dictionary untuk menyimpan semua metrics
    results = {
        'Model': [],
        'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-Score': [],
        'AUC-ROC': [],
        'TP': [], 'TN': [], 'FP': [], 'FN': [],
        'Validation_Acc': [], 'CrossVal_Mean': [], 'CrossVal_Std': []
    }
    
    predictions = {}
    confusion_matrices = {}
    feature_importances = {}
    
    print("\n" + "-"*60)
    
    for name, model in models.items():
        print(f"\n📊 {name}")
        print("-"*40)
        
        # 3.1 TRAIN MODEL
        print("   Training...")
        model.fit(X_train_scaled, y_train_encoded)
        
        # 3.2 PREDICT ON VALIDATION SET
        y_val_pred = model.predict(X_val_scaled)
        val_accuracy = accuracy_score(y_val_encoded, y_val_pred)
        
        # 3.3 PREDICT ON TEST SET
        y_test_pred = model.predict(X_test_scaled)
        y_test_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
        
        # 3.4 CALCULATE ALL METRICS
        accuracy = accuracy_score(y_test_encoded, y_test_pred)
        precision = precision_score(y_test_encoded, y_test_pred, average='weighted')
        recall = recall_score(y_test_encoded, y_test_pred, average='weighted')
        f1 = f1_score(y_test_encoded, y_test_pred, average='weighted')
        
        # AUC-ROC (jika ada probability predictions)
        auc_roc = 0
        if y_test_proba is not None:
            try:
                auc_roc = roc_auc_score(y_test_encoded, y_test_proba, multi_class='ovr', average='weighted')
            except:
                auc_roc = 0
        
        # Confusion Matrix dan TP/TN/FP/FN
        cm = confusion_matrix(y_test_encoded, y_test_pred)
        confusion_matrices[name] = cm
        
        # Calculate TP, TN, FP, FN per class lalu average
        n_classes = len(le.classes_)
        TP_sum, TN_sum, FP_sum, FN_sum = 0, 0, 0, 0
        
        for i in range(n_classes):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - (TP + FP + FN)
            
            TP_sum += TP
            TN_sum += TN
            FP_sum += FP
            FN_sum += FN
        
        # Average per class
        avg_TP = TP_sum / n_classes
        avg_TN = TN_sum / n_classes
        avg_FP = FP_sum / n_classes
        avg_FN = FN_sum / n_classes
        
        # 3.5 CROSS-VALIDATION
        print("   Cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_train_scaled, y_train_encoded, 
                                   cv=cv, scoring='accuracy')
        
        # 3.6 FEATURE IMPORTANCE (jika ada)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importances[name] = {
                'features': feature_cols,
                'importances': importances
            }
        
        # 3.7 STORE RESULTS
        results['Model'].append(name)
        results['Accuracy'].append(accuracy)
        results['Precision'].append(precision)
        results['Recall'].append(recall)
        results['F1-Score'].append(f1)
        results['AUC-ROC'].append(auc_roc)
        results['TP'].append(avg_TP)
        results['TN'].append(avg_TN)
        results['FP'].append(avg_FP)
        results['FN'].append(avg_FN)
        results['Validation_Acc'].append(val_accuracy)
        results['CrossVal_Mean'].append(cv_scores.mean())
        results['CrossVal_Std'].append(cv_scores.std())
        
        predictions[name] = {
            'y_true': y_test_encoded,
            'y_pred': y_test_pred,
            'y_proba': y_test_proba
        }
        
        # 3.8 PRINT RESULTS
        print(f"   ✅ Test Accuracy:    {accuracy:.4f}")
        print(f"   ✅ Validation Acc:   {val_accuracy:.4f}")
        print(f"   ✅ Precision:        {precision:.4f}")
        print(f"   ✅ Recall:           {recall:.4f}")
        print(f"   ✅ F1-Score:         {f1:.4f}")
        if auc_roc > 0:
            print(f"   ✅ AUC-ROC:          {auc_roc:.4f}")
        print(f"   📊 CV Mean ± Std:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"   📈 TP: {avg_TP:.1f}, TN: {avg_TN:.1f}, FP: {avg_FP:.1f}, FN: {avg_FN:.1f}")
        
        # 3.9 SAVE MODEL
        model_file = f"models_hard/{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, model_file)
        print(f"   💾 Model saved: {model_file}")
    
    print("\n" + "-"*60)
    
    # 4. CREATE RESULTS DATAFRAME
    print("\n📈 4. Creating comprehensive results comparison...")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    # Save detailed results
    results_df.to_csv('models_hard/model_comparison_detailed.csv', index=False)
    print("   💾 Detailed results saved: models_hard/model_comparison_detailed.csv")
    
    # 5. DISPLAY FINAL COMPARISON
    print("\n" + "="*70)
    print("FINAL MODEL COMPARISON - HARD TASK")
    print("="*70)
    
    display_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 
                   'AUC-ROC', 'Validation_Acc', 'CrossVal_Mean', 'CrossVal_Std']
    print(results_df[display_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # 6. CREATE VISUALIZATIONS
    print("\n🎨 5. Creating visualizations...")
    create_visualizations(results_df, confusion_matrices, feature_importances, le.classes_)
    
    # 7. DETERMINE BEST MODEL
    print("\n🏆 6. Determining best model...")
    
    # Cari best berdasarkan weighted score
    results_df['Weighted_Score'] = (
        results_df['Accuracy'] * 0.3 +
        results_df['F1-Score'] * 0.3 +
        results_df['Validation_Acc'] * 0.2 +
        results_df['CrossVal_Mean'] * 0.2
    )
    
    best_idx = results_df['Weighted_Score'].idxmax()
    best_model = results_df.loc[best_idx]
    
    print(f"\n   🥇 BEST OVERALL MODEL: {best_model['Model']}")
    print(f"   📊 Weighted Score:    {best_model['Weighted_Score']:.4f}")
    print(f"   📈 Test Accuracy:     {best_model['Accuracy']:.4f}")
    print(f"   📈 F1-Score:          {best_model['F1-Score']:.4f}")
    print(f"   📈 Validation Acc:    {best_model['Validation_Acc']:.4f}")
    
    # 8. DETAILED CLASSIFICATION REPORTS
    print("\n📋 7. Detailed classification reports...")
    
    for name in models.keys():
        print(f"\n{'='*50}")
        print(f"CLASSIFICATION REPORT - {name}")
        print('='*50)
        
        y_true = predictions[name]['y_true']
        y_pred = predictions[name]['y_pred']
        
        report = classification_report(y_true, y_pred, target_names=le.classes_)
        print(report)
        
        # Save report
        with open(f'models_hard/{name.lower().replace(" ", "_")}_report.txt', 'w') as f:
            f.write(f"Classification Report - {name}\n")
            f.write("="*60 + "\n\n")
            f.write(report)
    
    # 9. FINAL SUMMARY
    print("\n" + "="*70)
    print("TRAINING SUMMARY - HARD TASK")
    print("="*70)
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Task: Predict next period state (3 periods ahead)")
    print(f"📈 Best Model: {best_model['Model']} (Accuracy: {best_model['Accuracy']:.4f})")
    print(f"🎯 Realism Level: {'Excellent' if best_model['Accuracy'] < 0.9 else 'Good'}")
    
    print(f"\n💾 Files saved in 'models_hard/' folder:")
    print(f"   1. 3 trained models (.pkl)")
    print(f"   2. Model comparison CSV")
    print(f"   3. Feature scaler & label encoder")
    print(f"   4. 6 visualization charts")
    print(f"   5. 3 classification reports")
    print("="*70)
    
    return results_df, predictions, best_model

def create_visualizations(results_df, confusion_matrices, feature_importances, class_names):
    """Create 6 types of visualizations"""
    
    # Buat folder
    os.makedirs('models_hard/plots', exist_ok=True)
    
    # 1. METRICS COMPARISON BAR CHART
    fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))
    fig1.suptitle('Model Performance Metrics - Hard Task', fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Validation_Acc']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A0572', '#1A936F']
    
    for idx, (metric, color) in enumerate(zip(metrics, colors)):
        row = idx // 3
        col = idx % 3
        
        bars = axes1[row, col].bar(results_df['Model'], results_df[metric], color=color, alpha=0.8)
        axes1[row, col].set_title(f'{metric} Comparison', fontweight='bold')
        axes1[row, col].set_ylabel('Score')
        axes1[row, col].set_ylim([0, 1.05])
        axes1[row, col].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes1[row, col].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('models_hard/plots/1_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print("   ✅ 1. Metrics comparison chart saved")
    
    # 2. CONFUSION MATRICES HEATMAP
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    fig2.suptitle('Confusion Matrices - Hard Task', fontsize=16, fontweight='bold')
    
    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes2[idx], cbar_kws={'shrink': 0.8})
        axes2[idx].set_title(f'{name}', fontweight='bold')
        axes2[idx].set_xlabel('Predicted')
        axes2[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('models_hard/plots/2_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("   ✅ 2. Confusion matrices saved")
    
    # 3. TP/TN/FP/FN COMPARISON
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(results_df))
    width = 0.2
    
    ax3.bar(x - 1.5*width, results_df['TP'], width, label='True Positive', color='green', alpha=0.8)
    ax3.bar(x - 0.5*width, results_df['TN'], width, label='True Negative', color='blue', alpha=0.8)
    ax3.bar(x + 0.5*width, results_df['FP'], width, label='False Positive', color='orange', alpha=0.8)
    ax3.bar(x + 1.5*width, results_df['FN'], width, label='False Negative', color='red', alpha=0.8)
    
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Average Count per Class')
    ax3.set_title('True/False Positive/Negative Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(results_df['Model'], rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models_hard/plots/3_tp_tn_fp_fn.png', dpi=300, bbox_inches='tight')
    print("   ✅ 3. TP/TN/FP/FN chart saved")
    
    # 4. PERFORMANCE TREND LINE CHART
    fig4, ax4 = plt.subplots(figsize=(12, 8))
    
    metrics_line = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    markers = ['o', 's', '^', 'D']
    colors_line = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for metric, marker, color in zip(metrics_line, markers, colors_line):
        ax4.plot(results_df['Model'], results_df[metric], 
                marker=marker, markersize=10, linewidth=3,
                label=metric, color=color)
    
    ax4.set_xlabel('Models')
    ax4.set_ylabel('Score')
    ax4.set_title('Performance Metrics Trend', fontweight='bold')
    ax4.set_ylim([0, 1.05])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('models_hard/plots/4_performance_trend.png', dpi=300, bbox_inches='tight')
    print("   ✅ 4. Performance trend line chart saved")
    
    # 5. CROSS-VALIDATION RESULTS
    fig5, ax5 = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(results_df))
    
    # Plot CV mean dengan error bars
    bars = ax5.bar(x_pos, results_df['CrossVal_Mean'], 
                  yerr=results_df['CrossVal_Std'],
                  capsize=10, alpha=0.8, color='#6A0572')
    
    ax5.set_xlabel('Models')
    ax5.set_ylabel('Cross-Validation Accuracy')
    ax5.set_title('5-Fold Cross-Validation Results', fontweight='bold')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(results_df['Model'], rotation=45)
    ax5.set_ylim([0, 1.05])
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean, std in zip(bars, results_df['CrossVal_Mean'], results_df['CrossVal_Std']):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{mean:.3f} ± {std:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('models_hard/plots/5_cross_validation.png', dpi=300, bbox_inches='tight')
    print("   ✅ 5. Cross-validation chart saved")
    
    # 6. FEATURE IMPORTANCE (untuk model yang punya)
    if feature_importances:
        fig6, axes6 = plt.subplots(1, len(feature_importances), figsize=(18, 6))
        if len(feature_importances) == 1:
            axes6 = [axes6]
        
        fig6.suptitle('Feature Importance - Hard Task', fontsize=16, fontweight='bold')
        
        for idx, (name, imp_data) in enumerate(feature_importances.items()):
            if idx < len(axes6):
                # Sort features by importance
                sorted_idx = np.argsort(imp_data['importances'])[-10:]  # Top 10
                sorted_features = [imp_data['features'][i] for i in sorted_idx]
                sorted_importances = [imp_data['importances'][i] for i in sorted_idx]
                
                axes6[idx].barh(range(len(sorted_features)), sorted_importances, align='center')
                axes6[idx].set_yticks(range(len(sorted_features)))
                axes6[idx].set_yticklabels(sorted_features)
                axes6[idx].set_xlabel('Importance')
                axes6[idx].set_title(f'{name}')
                axes6[idx].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('models_hard/plots/6_feature_importance.png', dpi=300, bbox_inches='tight')
        print("   ✅ 6. Feature importance chart saved")
    
    print("   🎨 All 6 visualizations created successfully!")

def main():
    """Main function"""
    # Buat folder
    os.makedirs('models_hard', exist_ok=True)
    os.makedirs('models_hard/plots', exist_ok=True)
    
    print("🚀 HARD TASK MODEL TRAINING")
    print("="*60)
    print("Task: Predict next period's state (3 periods ahead)")
    print("Models: Decision Tree, KNN, Logistic Regression")
    print("="*60)
    
    # Jalankan training
    results_df, predictions, best_model = train_and_evaluate_hard_task()
    
    # Tanya user untuk tampilkan plots
    show_plots = input("\nShow all visualizations? (y/n): ").lower()
    if show_plots == 'y':
        plt.show()

if __name__ == "__main__":
    # Check dependencies
    try:
        import sklearn
        print(f"✅ scikit-learn: {sklearn.__version__}")
    except ImportError:
        print("❌ Install: pip install scikit-learn")
        exit(1)
    
    try:
        import matplotlib
        print(f"✅ matplotlib: {matplotlib.__version__}")
    except ImportError:
        print("❌ Install: pip install matplotlib")
        exit(1)
    
    try:
        import seaborn
        print(f"✅ seaborn: {seaborn.__version__}")
    except ImportError:
        print("❌ Install: pip install seaborn")
        exit(1)
    
    main()