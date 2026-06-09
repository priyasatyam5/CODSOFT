"""
============================================================
  CREDIT CARD FRAUD DETECTION — Complete ML Pipeline
  Internship Project | Kartik2112 Kaggle Dataset
============================================================

OVERVIEW
--------
This script trains and evaluates 3 models:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest

It handles the key challenge of fraud detection:
  CLASS IMBALANCE — fraud is very rare (~0.5% of transactions).

HOW TO USE
----------
1. Download dataset from:
   https://www.kaggle.com/datasets/kartik2112/fraud-detection
2. Place fraudTrain.csv and fraudTest.csv in the same folder
3. Run: python fraud_detection.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score,
    precision_score, recall_score, accuracy_score
)
from sklearn.utils import resample

# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 1: LOADING DATA")
print("="*60)

# Try to load real dataset; fall back to synthetic demo data
try:
    train_df = pd.read_csv('fraudTrain.csv')
    test_df  = pd.read_csv('fraudTest.csv')
    print(f"✅ Loaded real dataset")
    print(f"   Train: {train_df.shape[0]:,} rows | Test: {test_df.shape[0]:,} rows")
    SYNTHETIC = False

except FileNotFoundError:
    print("⚠️  Dataset files not found. Generating synthetic demo data...")
    print("   Download the real dataset from Kaggle for actual internship use.")

    # ── Synthetic data generator ─────────────────────────────
    np.random.seed(42)
    N_TRAIN, N_TEST = 80_000, 20_000

    def make_dataset(n, fraud_frac=0.006):
        n_fraud = int(n * fraud_frac)
        n_legit = n - n_fraud

        legit = {
            'amt':        np.random.lognormal(3.5, 1.0, n_legit),
            'city_pop':   np.random.randint(1_000, 3_000_000, n_legit),
            'hour':       np.random.randint(6, 23, n_legit),
            'category':   np.random.choice(
                              ['grocery_pos','gas_transport','home','shopping_net',
                               'entertainment','food_dining','health_fitness',
                               'misc_net','shopping_pos','misc_pos'], n_legit),
            'gender':     np.random.choice(['M','F'], n_legit),
            'age':        np.random.randint(18, 80, n_legit),
            'is_fraud':   np.zeros(n_legit, dtype=int)
        }

        fraud = {
            'amt':        np.random.lognormal(5.5, 1.2, n_fraud),   # higher amounts
            'city_pop':   np.random.randint(1_000, 500_000, n_fraud),
            'hour':       np.random.choice([0,1,2,3,22,23], n_fraud),  # odd hours
            'category':   np.random.choice(
                              ['misc_net','shopping_net','grocery_pos'], n_fraud),
            'gender':     np.random.choice(['M','F'], n_fraud),
            'age':        np.random.randint(18, 80, n_fraud),
            'is_fraud':   np.ones(n_fraud, dtype=int)
        }

        df = pd.concat([pd.DataFrame(legit), pd.DataFrame(fraud)], ignore_index=True)
        return df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df = make_dataset(N_TRAIN)
    test_df  = make_dataset(N_TEST)
    SYNTHETIC = True
    print(f"   Train: {train_df.shape[0]:,} rows | Test: {test_df.shape[0]:,} rows")


# ─────────────────────────────────────────────
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 2: EXPLORATORY DATA ANALYSIS")
print("="*60)

fraud_count = train_df['is_fraud'].value_counts()
fraud_pct   = train_df['is_fraud'].mean() * 100
print(f"\nClass Distribution in Training Set:")
print(f"   Legitimate:  {fraud_count.get(0, 0):>10,}  ({100-fraud_pct:.2f}%)")
print(f"   Fraudulent:  {fraud_count.get(1, 0):>10,}  ({fraud_pct:.4f}%)")
print(f"\n⚠️  SEVERE CLASS IMBALANCE — fraud is only {fraud_pct:.2f}% of all transactions")
print("   This is realistic: in real life, fraud is extremely rare.")
print("   Without handling this, models just predict 'legitimate' for everything!")

print(f"\nDataset Columns: {list(train_df.columns)}")
print(f"\nSample rows:\n{train_df.head(3).to_string()}")


# ─────────────────────────────────────────────
# STEP 3: FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 3: FEATURE ENGINEERING")
print("="*60)
print("""
Feature Engineering = creating new useful columns from raw data.

For example:
  • Extract 'hour' from transaction timestamp (fraud often at odd hours)
  • Age of customer  
  • Distance between customer home and merchant
  • Transaction amount (fraud tends to be high or unusually low)
""")

def engineer_features(df):
    """Create ML-ready features from raw transaction data."""
    df = df.copy()

    # ── Handle real Kaggle dataset columns ────────────────────
    if 'trans_date_trans_time' in df.columns:
        df['trans_date_trans_time'] = pd.to_datetime(
            df['trans_date_trans_time'], errors='coerce')
        df['hour']    = df['trans_date_trans_time'].dt.hour
        df['day']     = df['trans_date_trans_time'].dt.dayofweek
        df['month']   = df['trans_date_trans_time'].dt.month

    if 'dob' in df.columns:
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['age'] = (pd.Timestamp.now() - df['dob']).dt.days // 365

    # Distance: merchant vs home coordinates
    if all(c in df.columns for c in ['merch_lat','merch_long','lat','long']):
        df['distance'] = np.sqrt(
            (df['merch_lat'] - df['lat'])**2 +
            (df['merch_long'] - df['long'])**2
        )

    # ── Select numeric + categorical features ─────────────────
    numeric_cols = ['amt', 'city_pop', 'hour']
    for col in ['age', 'day', 'month', 'distance', 'unix_time']:
        if col in df.columns:
            numeric_cols.append(col)

    categorical_cols = []
    for col in ['category', 'gender']:
        if col in df.columns:
            categorical_cols.append(col)

    # Label-encode categorical columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col + '_enc'] = le.fit_transform(df[col].astype(str))

    encoded_cats = [c + '_enc' for c in categorical_cols]
    feature_cols = numeric_cols + encoded_cats

    # Keep only available features
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Fill missing values
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    return df, feature_cols

train_df, FEATURES = engineer_features(train_df)
test_df,  _        = engineer_features(test_df)

print(f"Features used for training: {FEATURES}")
print(f"Number of features: {len(FEATURES)}")


# ─────────────────────────────────────────────
# STEP 4: HANDLE CLASS IMBALANCE
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 4: HANDLING CLASS IMBALANCE (Undersampling)")
print("="*60)
print("""
PROBLEM: If 99.5% of data is 'not fraud', a dumb model that always 
         predicts 'not fraud' gets 99.5% accuracy — but catches ZERO fraud!

SOLUTION: Balance the classes before training.

Methods:
  1. UNDERSAMPLING  — reduce majority class (we use this, fast & simple)
  2. OVERSAMPLING   — duplicate minority class (SMOTE is popular)
  3. Class Weights  — penalize wrong predictions on minority class more

We use undersampling: match legitimate count to fraud count.
""")

fraud_train  = train_df[train_df['is_fraud'] == 1]
legit_train  = train_df[train_df['is_fraud'] == 0]

# Undersample legitimate to 5× the fraud count (keeps more data than 1:1)
target_legit = min(len(fraud_train) * 5, len(legit_train))
legit_down   = resample(legit_train, n_samples=target_legit, random_state=42)
balanced_df  = pd.concat([fraud_train, legit_down]).sample(frac=1, random_state=42)

print(f"Before balancing: Fraud={len(fraud_train):,} | Legit={len(legit_train):,}")
print(f"After balancing:  Fraud={len(fraud_train):,} | Legit={len(legit_down):,}  (5:1 ratio)")

X_train = balanced_df[FEATURES].values
y_train = balanced_df['is_fraud'].values
X_test  = test_df[FEATURES].values
y_test  = test_df['is_fraud'].values


# ─────────────────────────────────────────────
# STEP 5: SCALE FEATURES
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 5: FEATURE SCALING")
print("="*60)
print("""
Logistic Regression needs features on the same scale.
StandardScaler transforms each feature to mean=0, std=1.
Decision Tree & Random Forest don't need scaling, but it doesn't hurt.
""")

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit on train ONLY
X_test  = scaler.transform(X_test)        # apply same scale to test


# ─────────────────────────────────────────────
# STEP 6: TRAIN MODELS
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 6: TRAINING MODELS")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, C=1.0, random_state=42,
        # class_weight='balanced' also works instead of undersampling
    ),
    'Decision Tree': DecisionTreeClassifier(
        max_depth=8, min_samples_leaf=10, random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=5,
        n_jobs=-1, random_state=42
    )
}

print("""
Model explanations:

1. LOGISTIC REGRESSION
   Simple, fast, interpretable. Finds a linear boundary between classes.
   Good baseline. Works well when features have linear relationships.

2. DECISION TREE
   Asks yes/no questions about features to classify.
   E.g., "Is amount > $500?" → "Is hour < 3 AM?" → "FRAUD"
   Interpretable, but can overfit.

3. RANDOM FOREST
   100 decision trees, each trained on a random subset of data & features.
   Final prediction = majority vote of all trees.
   Very powerful for fraud detection — our best model!
""")

results     = {}
predictions = {}

for name, model in models.items():
    print(f"Training {name}... ", end='', flush=True)
    model.fit(X_train, y_train)
    y_pred     = model.predict(X_test)
    y_prob     = model.predict_proba(X_test)[:, 1]
    auc        = roc_auc_score(y_test, y_prob)
    f1         = f1_score(y_test, y_pred)
    precision  = precision_score(y_test, y_pred, zero_division=0)
    recall     = recall_score(y_test, y_pred)
    accuracy   = accuracy_score(y_test, y_pred)
    cm         = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    results[name] = {
        'accuracy':  accuracy,
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'auc':       auc,
        'cm':        cm.tolist(),
        'fpr':       fpr.tolist(),
        'tpr':       tpr.tolist(),
        'y_prob':    y_prob.tolist()
    }
    predictions[name] = (y_pred, y_prob)
    print(f"Done! AUC={auc:.4f} | F1={f1:.4f} | Recall={recall:.4f}")


# ─────────────────────────────────────────────
# STEP 7: EVALUATE & COMPARE
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 7: MODEL EVALUATION")
print("="*60)
print("""
WHY NOT USE ACCURACY?
  With 0.5% fraud rate, even a model predicting ALL legitimate gets 99.5% accuracy.
  That model is USELESS for fraud detection!

BETTER METRICS FOR IMBALANCED DATA:
  • PRECISION  = Of all transactions flagged as fraud, how many are actually fraud?
                 High precision → fewer false alarms (good for customer experience)
  • RECALL     = Of all actual frauds, how many did we catch?
                 High recall → fewer frauds missed (critical for the bank!)
  • F1 SCORE   = Harmonic mean of precision & recall (balance of both)
  • AUC-ROC    = Area under ROC curve; 1.0 = perfect, 0.5 = random guess
  • Confusion Matrix = TP, FP, TN, FN breakdown
""")

print(f"\n{'Model':<25} {'Accuracy':>9} {'Precision':>9} {'Recall':>9} {'F1':>9} {'AUC':>9}")
print("-" * 65)
for name, r in results.items():
    print(f"{name:<25} {r['accuracy']:>9.4f} {r['precision']:>9.4f} {r['recall']:>9.4f} {r['f1']:>9.4f} {r['auc']:>9.4f}")

best_model = max(results, key=lambda k: results[k]['auc'])
print(f"\n🏆 Best Model: {best_model} (AUC = {results[best_model]['auc']:.4f})")

for name, r in results.items():
    cm = np.array(r['cm'])
    tn, fp, fn, tp = cm.ravel()
    print(f"\n{name} Confusion Matrix:")
    print(f"   True Negatives  (correct legit):    {tn:>8,}")
    print(f"   False Positives (wrong fraud alert): {fp:>8,}")
    print(f"   False Negatives (missed fraud!):     {fn:>8,}")
    print(f"   True Positives  (caught fraud!):     {tp:>8,}")

# Feature importance from Random Forest
if 'Random Forest' in models:
    rf = models['Random Forest']
    importances = rf.feature_importances_
    feat_imp = sorted(zip(FEATURES, importances), key=lambda x: -x[1])
    print("\nRandom Forest — Feature Importances:")
    for feat, imp in feat_imp:
        bar = '█' * int(imp * 50)
        print(f"   {feat:<15} {bar} {imp:.4f}")


# ─────────────────────────────────────────────
# STEP 8: PLOTS
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("STEP 8: GENERATING VISUALIZATIONS")
print("="*60)

colors = {'Logistic Regression': '#3B82F6', 'Decision Tree': '#F59E0B', 'Random Forest': '#10B981'}
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.patch.set_facecolor('#0F172A')
for ax in axes.flat:
    ax.set_facecolor('#1E293B')

# ── Plot 1: Class Distribution ─────────────
ax = axes[0, 0]
labels = ['Legitimate', 'Fraudulent']
vals   = [fraud_count.get(0, 0), fraud_count.get(1, 0)]
bars   = ax.bar(labels, vals, color=['#3B82F6', '#EF4444'], width=0.5, edgecolor='none')
ax.set_title('Class Distribution', color='white', fontsize=13, pad=12)
ax.set_ylabel('Count', color='#94A3B8')
ax.tick_params(colors='#94A3B8')
for spine in ax.spines.values(): spine.set_visible(False)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
            f'{val:,}', ha='center', va='bottom', color='white', fontsize=10)
ax.set_facecolor('#1E293B')
ax.yaxis.label.set_color('#94A3B8')

# ── Plot 2: ROC Curves ──────────────────────
ax = axes[0, 1]
for name, r in results.items():
    ax.plot(r['fpr'], r['tpr'], color=colors[name], linewidth=2.5,
            label=f"{name} (AUC={r['auc']:.3f})")
ax.plot([0,1],[0,1], 'w--', alpha=0.3, linewidth=1)
ax.set_title('ROC Curves', color='white', fontsize=13, pad=12)
ax.set_xlabel('False Positive Rate', color='#94A3B8')
ax.set_ylabel('True Positive Rate', color='#94A3B8')
ax.tick_params(colors='#94A3B8')
ax.legend(facecolor='#0F172A', labelcolor='white', fontsize=9)
for spine in ax.spines.values(): spine.set_color('#334155')

# ── Plot 3: Metric Comparison ───────────────
ax = axes[0, 2]
metric_names = ['Precision', 'Recall', 'F1', 'AUC']
x = np.arange(len(metric_names))
w = 0.25
for i, (name, r) in enumerate(results.items()):
    vals_m = [r['precision'], r['recall'], r['f1'], r['auc']]
    ax.bar(x + i*w - w, vals_m, w, label=name, color=colors[name], alpha=0.9)
ax.set_title('Model Comparison', color='white', fontsize=13, pad=12)
ax.set_xticks(x); ax.set_xticklabels(metric_names, color='#94A3B8')
ax.tick_params(colors='#94A3B8')
ax.set_ylim(0, 1.1)
ax.legend(facecolor='#0F172A', labelcolor='white', fontsize=9)
for spine in ax.spines.values(): spine.set_color('#334155')

# ── Plot 4-6: Confusion Matrices ────────────
for idx, (name, r) in enumerate(results.items()):
    ax = axes[1, idx]
    cm = np.array(r['cm'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap='Blues', cbar=False,
                annot_kws={'color': 'white', 'size': 12})
    ax.set_title(f'{name}\nConfusion Matrix', color='white', fontsize=11, pad=8)
    ax.set_xlabel('Predicted', color='#94A3B8')
    ax.set_ylabel('Actual', color='#94A3B8')
    ax.set_xticklabels(['Legit', 'Fraud'], color='#94A3B8')
    ax.set_yticklabels(['Legit', 'Fraud'], color='#94A3B8', rotation=0)
    for spine in ax.spines.values(): spine.set_color('#334155')

plt.suptitle('Credit Card Fraud Detection — Model Evaluation',
             color='white', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('fraud_detection_results.png',
            dpi=150, bbox_inches='tight', facecolor='#0F172A')
print("✅ Saved: fraud_detection_results.png")

# ─────────────────────────────────────────────
# STEP 9: SAVE RESULTS JSON for dashboard
# ─────────────────────────────────────────────
summary = {
    'synthetic': SYNTHETIC,
    'dataset_size': {'train': len(train_df), 'test': len(test_df)},
    'fraud_rate_pct': round(fraud_pct, 4),
    'features': FEATURES,
    'models': {
        name: {
            'accuracy':  round(r['accuracy'], 4),
            'precision': round(r['precision'], 4),
            'recall':    round(r['recall'], 4),
            'f1':        round(r['f1'], 4),
            'auc':       round(r['auc'], 4),
            'cm':        r['cm'],
            'fpr':       [round(x,4) for x in r['fpr'][::max(1,len(r['fpr'])//50)]],
            'tpr':       [round(x,4) for x in r['tpr'][::max(1,len(r['tpr'])//50)]]
        }
        for name, r in results.items()
    },
    'best_model': best_model,
    'feature_importances': (
        {f: round(float(imp), 4)
         for f, imp in zip(FEATURES, models['Random Forest'].feature_importances_)}
        if 'Random Forest' in models else {}
    )
}

with open('results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("✅ Saved: results.json")

print("\n" + "="*60)
print("✅ FRAUD DETECTION PIPELINE COMPLETE!")
print("="*60)
print(f"\n  Best Model : {best_model}")
print(f"  AUC Score  : {results[best_model]['auc']:.4f}")
print(f"  F1 Score   : {results[best_model]['f1']:.4f}")
print(f"  Recall     : {results[best_model]['recall']:.4f}")

