"""
=============================================================
 BANK CUSTOMER CHURN PREDICTION - INTERNSHIP PROJECT by SHANMUKHA PRIYA GANTYADA
 Algorithms: Logistic Regression, Random Forest, Gradient Boosting
=============================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.inspection import permutation_importance

# ─────────────────────────────────────────────
# 1. LOAD & EXPLORE DATA
# ─────────────────────────────────────────────
print("=" * 60)
print(" STEP 1: LOADING AND EXPLORING DATA")
print("=" * 60)

df = pd.read_csv('Churn_Modelling.csv')
print(f"\nShape: {df.shape}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nTarget Distribution:\n{df['Exited'].value_counts()}")
print(f"Churn Rate: {df['Exited'].mean():.2%}")
print(f"\nNumerical Summary:\n{df.describe().round(2)}")

# ─────────────────────────────────────────────
# 2. DATA PREPROCESSING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(" STEP 2: PREPROCESSING")
print("=" * 60)

# Drop non-predictive columns
df_model = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode categorical features
le_geo = LabelEncoder()
le_gen = LabelEncoder()
df_model['Geography'] = le_geo.fit_transform(df_model['Geography'])
df_model['Gender'] = le_gen.fit_transform(df_model['Gender'])

# Feature / target split
X = df_model.drop('Exited', axis=1)
y = df_model['Exited']

feature_names = X.columns.tolist()
print(f"Features: {feature_names}")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Train / test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
print(f"Train churn rate: {y_train.mean():.2%} | Test churn rate: {y_test.mean():.2%}")

# Scale features
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Compute class weights
classes = np.array([0, 1])
cw = compute_class_weight('balanced', classes=classes, y=y_train)
class_weights = {0: cw[0], 1: cw[1]}
print(f"Class weights: {class_weights}")

# ─────────────────────────────────────────────
# 3. MODEL TRAINING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(" STEP 3: TRAINING MODELS")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, class_weight='balanced', random_state=42, C=0.1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, random_state=42
    )
}

# ─────────────────────────────────────────────
# 4. EVALUATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(" STEP 4: EVALUATION")
print("=" * 60)

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")

    # Use scaled data for LR, raw for tree-based
    Xtr = X_train_sc if name == "Logistic Regression" else X_train
    Xte = X_test_sc  if name == "Logistic Regression" else X_test
    Xtr_cv = X_train_sc if name == "Logistic Regression" else np.array(X_train)

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    f1   = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    ap   = average_precision_score(y_test, y_prob)

    cv_scores = cross_val_score(model, Xtr_cv, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'accuracy': acc,
        'roc_auc': auc,
        'f1': f1,
        'precision': prec,
        'recall': rec,
        'avg_precision': ap,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'Xte': Xte
    }

    print(f"  Accuracy:        {acc:.4f}")
    print(f"  ROC-AUC:         {auc:.4f}")
    print(f"  F1 Score:        {f1:.4f}")
    print(f"  Precision:       {prec:.4f}")
    print(f"  Recall:          {rec:.4f}")
    print(f"  Avg Precision:   {ap:.4f}")
    print(f"  CV AUC (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred, target_names=['Stayed','Churned'])}")

# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────
rf_importances = results["Random Forest"]["model"].feature_importances_
feature_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': rf_importances
}).sort_values('Importance', ascending=False)

print("\n" + "=" * 60)
print(" FEATURE IMPORTANCES (Random Forest)")
print("=" * 60)
print(feature_imp_df.to_string(index=False))

# ─────────────────────────────────────────────
# 6. SAVE METRICS JSON FOR DASHBOARD
# ─────────────────────────────────────────────
metrics_out = {}
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, r['y_prob'])
    cm = confusion_matrix(y_test, r['y_pred'])
    metrics_out[name] = {
        'accuracy':      round(r['accuracy'], 4),
        'roc_auc':       round(r['roc_auc'], 4),
        'f1':            round(r['f1'], 4),
        'precision':     round(r['precision'], 4),
        'recall':        round(r['recall'], 4),
        'avg_precision': round(r['avg_precision'], 4),
        'cv_mean':       round(r['cv_mean'], 4),
        'cv_std':        round(r['cv_std'], 4),
        'fpr':           fpr.tolist()[::5],
        'tpr':           tpr.tolist()[::5],
        'pr_precision':  prec_arr.tolist()[::5],
        'pr_recall':     rec_arr.tolist()[::5],
        'cm':            cm.tolist()
    }

with open('metrics.json', 'w') as f:
    json.dump({
        'models': metrics_out,
        'feature_importance': feature_imp_df.to_dict(orient='records'),
        'churn_rate': float(df['Exited'].mean()),
        'n_samples': len(df),
        'n_churned': int(df['Exited'].sum()),
        'geo_labels': le_geo.classes_.tolist(),
    }, f, indent=2)

print("\nMetrics saved to metrics.json")

# ─────────────────────────────────────────────
# 7. GENERATE PLOTS
# ─────────────────────────────────────────────
colors = {'Logistic Regression': '#6366f1', 'Random Forest': '#10b981', 'Gradient Boosting': '#f59e0b'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.dpi': 130})

# ── Plot 1: EDA Overview ──
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('Exploratory Data Analysis — Bank Customer Churn', fontsize=15, fontweight='bold', y=1.01)

ax = axes[0,0]
churn_counts = df['Exited'].value_counts()
bars = ax.bar(['Stayed', 'Churned'], churn_counts.values, color=['#10b981','#ef4444'], width=0.5, edgecolor='white', linewidth=2)
ax.set_title('Churn Distribution', fontweight='bold')
ax.set_ylabel('Count')
for bar, val in zip(bars, churn_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f'{val:,}\n({val/len(df)*100:.1f}%)', ha='center', fontsize=10)
ax.set_ylim(0, max(churn_counts)*1.2)

ax = axes[0,1]
df.groupby(['Geography','Exited']).size().unstack().plot(kind='bar', ax=ax, color=['#10b981','#ef4444'], edgecolor='white')
ax.set_title('Churn by Geography', fontweight='bold')
ax.set_xlabel('')
ax.set_xticklabels(['France','Germany','Spain'], rotation=0)
ax.legend(['Stayed','Churned'])

ax = axes[0,2]
df[df['Exited']==0]['Age'].hist(ax=ax, bins=30, alpha=0.6, color='#10b981', label='Stayed')
df[df['Exited']==1]['Age'].hist(ax=ax, bins=30, alpha=0.6, color='#ef4444', label='Churned')
ax.set_title('Age Distribution by Churn', fontweight='bold')
ax.set_xlabel('Age')
ax.legend()

ax = axes[1,0]
df.groupby(['NumOfProducts','Exited']).size().unstack().plot(kind='bar', ax=ax, color=['#10b981','#ef4444'], edgecolor='white')
ax.set_title('Products vs Churn', fontweight='bold')
ax.set_xlabel('Number of Products')
ax.set_xticklabels([1,2,3,4], rotation=0)
ax.legend(['Stayed','Churned'])

ax = axes[1,1]
df[df['Exited']==0]['CreditScore'].hist(ax=ax, bins=30, alpha=0.6, color='#10b981', label='Stayed')
df[df['Exited']==1]['CreditScore'].hist(ax=ax, bins=30, alpha=0.6, color='#ef4444', label='Churned')
ax.set_title('Credit Score Distribution', fontweight='bold')
ax.set_xlabel('Credit Score')
ax.legend()

ax = axes[1,2]
corr = df_model.corr()['Exited'].drop('Exited').sort_values()
clrs = ['#ef4444' if v > 0 else '#10b981' for v in corr.values]
ax.barh(corr.index, corr.values, color=clrs)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Feature Correlation with Churn', fontweight='bold')
ax.set_xlabel('Pearson r')

plt.tight_layout()
plt.savefig('(     )eda_overview.png', bbox_inches='tight', dpi=130)
plt.close()
print("Saved: eda_overview.png")

# ── Plot 2: ROC Curves ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Model Performance', fontsize=14, fontweight='bold')

ax = axes[0]
ax.plot([0,1],[0,1],'--', color='grey', lw=1, label='Random (AUC=0.50)')
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    ax.plot(fpr, tpr, color=colors[name], lw=2.2,
            label=f"{name} (AUC={r['roc_auc']:.3f})")
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves', fontweight='bold')
ax.legend(fontsize=9)
ax.set_facecolor('#f8fafc')

ax = axes[1]
for name, r in results.items():
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, r['y_prob'])
    ap = r['avg_precision']
    ax.plot(rec_arr, prec_arr, color=colors[name], lw=2.2,
            label=f"{name} (AP={ap:.3f})")
ax.axhline(y_test.mean(), color='grey', linestyle='--', lw=1, label=f'Baseline ({y_test.mean():.2f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves', fontweight='bold')
ax.legend(fontsize=9)
ax.set_facecolor('#f8fafc')

plt.tight_layout()
plt.savefig('(     )roc_pr_curves.png', bbox_inches='tight', dpi=130)
plt.close()
print("Saved: roc_pr_curves.png")

# ── Plot 3: Confusion Matrices ──
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Stayed','Churned'],
                yticklabels=['Stayed','Churned'],
                linewidths=2, linecolor='white',
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(name, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('(     )confusion_matrices.png', bbox_inches='tight', dpi=130)
plt.close()
print("Saved: confusion_matrices.png")

# ── Plot 4: Feature Importances ──
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Feature Analysis', fontsize=14, fontweight='bold')

ax = axes[0]
fi = feature_imp_df.head(10)
bars = ax.barh(fi['Feature'], fi['Importance'], color='#6366f1', edgecolor='white')
ax.set_title('Random Forest Feature Importances', fontweight='bold')
ax.set_xlabel('Importance Score')
ax.invert_yaxis()
for bar, val in zip(bars, fi['Importance']):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=9)

ax = axes[1]
metric_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [r['accuracy'] for r in results.values()],
    'ROC-AUC':  [r['roc_auc']  for r in results.values()],
    'F1 Score': [r['f1']       for r in results.values()],
    'Recall':   [r['recall']   for r in results.values()],
}).set_index('Model')
metric_df.plot(kind='bar', ax=ax, color=['#6366f1','#10b981','#f59e0b','#ef4444'],
               edgecolor='white', width=0.7)
ax.set_title('Model Comparison', fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.0)
ax.legend(loc='lower right', fontsize=9)
ax.axhline(0.8, color='black', linestyle='--', lw=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig('(     )feature_model_comparison.png', bbox_inches='tight', dpi=130)
plt.close()
print("Saved: feature_model_comparison.png")

print("\n✅ All outputs generated successfully!")
