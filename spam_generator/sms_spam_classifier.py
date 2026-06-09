"""
SMS Spam Classification - Complete ML Pipeline
===============================================
Dataset: UCI SMS Spam Collection
        https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset

This script trains and evaluates multiple classifiers:
  - Naive Bayes (MultinomialNB)
  - Logistic Regression
  - Support Vector Machine (LinearSVC)

Feature extraction:
  - TF-IDF (unigrams + bigrams)
  - Count Vectorizer (baseline)

Usage:
  python sms_spam_classifier.py                        # Uses built-in demo dataset
  python sms_spam_classifier.py --data spam.csv        # Use your Kaggle dataset
  python sms_spam_classifier.py --predict "Win a prize now!"
"""

import argparse
import os
import sys
import json
import warnings
import pickle
import time

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score
)
from sklearn.preprocessing import LabelEncoder
import re

# ─────────────────────────────────────────────
#  DEMO DATASET (used if no CSV is provided)
# ─────────────────────────────────────────────
DEMO_MESSAGES = [
    ("ham",  "Go until jurong point, crazy.. Available only in bugis n great world la e buffet"),
    ("spam", "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry"),
    ("ham",  "U dun say so early hor... U c already then say..."),
    ("spam", "WINNER!! As a valued network customer you have been selected to receive a 900 pound prize reward"),
    ("ham",  "Nah I don't think he goes to usf, he lives around here though"),
    ("spam", "SIX chances to win CASH! From 100 to 20,000 pounds txt CSH11 and send to 87575. Cost 150p/day"),
    ("ham",  "Even my brother is not like to speak with me. They treat me like aids patient"),
    ("spam", "URGENT! You have won a 1 week FREE membership in our 100,000 Prize Jackpot! Txt WIN to 80086"),
    ("ham",  "I've been searching for the right words to thank you for this breather"),
    ("spam", "XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message"),
    ("ham",  "I HAVE A DATE ON SUNDAY WITH WILL!!"),
    ("spam", "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera"),
    ("ham",  "Oh k...i'm watching here:)"),
    ("spam", "FREE! Talk, Text and Surf the internet from your phone HALF PRICE now 50p per month"),
    ("ham",  "Sian. I'm not sure I know the way there. Do you want me to be there?"),
    ("spam", "Congratulations! You've been selected to receive a 200 pound Argos voucher, no purchase required"),
    ("ham",  "Will u meet ur dream partner soon? Is ur career off 2 a flying start? 2 find out free reply YES"),
    ("spam", "URGENT! Your Mobile number has been awarded a 2000 Bonus Caller Prize on 5/9/03!"),
    ("ham",  "You are a great friend, your advice is always helpful."),
    ("spam", "Claim a free prize by texting YES to 80488 now!"),
    ("ham",  "Hey, are we still on for dinner tonight?"),
    ("ham",  "Ok I'll wait and see."),
    ("spam", "Your free ringtone is waiting to be collected. Simply text the password MIX to 85069"),
    ("ham",  "Sorry, I'll call later"),
    ("spam", "You have been selected as a WINNER of 500 pounds prize. Call 09061701461 from landline."),
    ("ham",  "Lovely day today isn't it?"),
    ("ham",  "On my way, be there in 20 mins."),
    ("spam", "Win a year supply of chocolate! Text CHOCO to 87543 now!"),
    ("ham",  "Dinner at 8? Let me know if you can make it."),
    ("ham",  "Ok! Going home now."),
    ("spam", "CASH Prize! Get 200 pounds by calling 0906 282 7720 now from your mobile."),
    ("ham",  "Happy birthday! Hope you have a great day."),
    ("spam", "Urgent! You've won a free iPhone. Click here to claim your reward immediately."),
    ("ham",  "Are you free this weekend? Let's catch up."),
    ("ham",  "Just finished work, exhausted."),
    ("spam", "Congratulations ur awarded either 500 of CD vouchers or 125 gift guaranteed and free entry"),
    ("ham",  "I'm at the library, studying for exams."),
    ("ham",  "Can you pick me up from the station?"),
    ("spam", "WIN A NOKIA 7250i. Win 1 of 20 Nokia phones NOW. Txt NOKIA to 86021."),
    ("ham",  "Running late, will be there at 7."),
    ("ham",  "The meeting is at 10 AM tomorrow."),
    ("spam", "Call 0800 to receive your FREE gift. No obligation required."),
    ("ham",  "Thanks for letting me know."),
    ("spam", "Act now! Limited time offer. Call 0906 200 3900 from your mobile."),
    ("ham",  "See you at lunch?"),
    ("spam", "You are a winner! Click to claim £1000 reward before midnight!"),
    ("ham",  "I'll be home by 9. Save some dinner for me."),
    ("spam", "FREE msg: You've won 2 tickets to the FA Cup Final worth £200. Call 09100 9121 75."),
    ("ham",  "What time does your flight leave?"),
    ("spam", "Your account is credited with 5000 free SMS. Send Y to claim."),
    ("ham",  "Forgot to mention, mom called earlier."),
    ("spam", "Congratulations! Nokia 3650 phone is your lucky draw prize. To claim call 09064012103"),
    ("ham",  "Can you bring the notes from last week's class?"),
    ("spam", "Latest Nokia phones for you FREE. Call 0800 056 5859 for a free phone."),
    ("ham",  "I'm staying at Shelly's place tonight."),
    ("spam", "Double your salary! Text SALARY to 89555 now for details."),
    ("ham",  "Just saw your message, calling you back now."),
    ("spam", "IMPORTANT: Your SIM card expires today. Call 0800 123 456 to renew for FREE."),
    ("ham",  "Let's grab coffee after class tomorrow."),
    ("spam", "Earn 150 per hour as a mystery shopper. No experience needed. Text SHOP to 63735."),
    ("ham",  "I think I left my jacket at your place."),
    ("spam", "You've been chosen as part of our customer survey for a 500 pound reward."),
    ("ham",  "Alright, let's meet at the usual spot."),
    ("spam", "FREE ringtones and wallpapers for your mobile! Text FREE to 87070"),
    ("ham",  "That sounds good to me, I'll be there."),
    ("spam", "Reply WIN to claim your 1000 prize today. Offer expires midnight."),
    ("ham",  "Did you finish the assignment?"),
    ("ham",  "It's been a long day. Can't wait for the weekend."),
    ("spam", "TEXT back STOP to opt out of these amazing deals and free prizes"),
    ("ham",  "Don't forget we have a game tonight."),
    ("spam", "Mobile number selected for prize. Call 08715203656 to claim your reward."),
    ("ham",  "Are you watching the match tonight?"),
    ("ham",  "Just got back from the gym. Totally wiped out."),
    ("spam", "U win a brand new iPod. Text WIN to 85023 for free entry."),
    ("ham",  "Call me when you get a chance."),
    ("spam", "Todays Offer: Get 60% cashback on all purchases. Limited time. Click now."),
    ("ham",  "Hope you're feeling better today."),
    ("spam", "Jackpot! You've won the weekly lottery. Call 09064019788 to collect."),
    ("ham",  "Let me know if you need anything from the store."),
    ("spam", "Secret codes to win FREE prizes sent to your mobile. Text PRIZE to 80082"),
    ("ham",  "Are you coming to the party on Saturday?"),
    ("ham",  "Finished the book, it was great!"),
    ("spam", "Exclusive Deal for you: Pay 0 and get unlimited calls. Text YES to 88800."),
    ("ham",  "Good morning! Have a wonderful day."),
    ("spam", "You qualify for a FREE home energy upgrade. No cost to you. Call 0800 234 567."),
    ("ham",  "Pick you up at 6?"),
    ("spam", "Your opinion matters! Complete our survey and WIN 200 shopping vouchers."),
    ("ham",  "I'll take care of it tonight."),
    ("ham",  "See you on Monday."),
    ("spam", "ALERT: Your account has been compromised. Call 0800 987 654 now."),
    ("ham",  "I just got promoted!"),
    ("spam", "Cheap international calls from 1p/min. Text CHEAP to 87070"),
    ("ham",  "What's for lunch today?"),
    ("spam", "You are a VIP customer. Your reward of 500 cash is ready to collect."),
    ("ham",  "Just stepped out for a bit, be back soon."),
    ("spam", "FREE iPhone 15 for survey completion. Click to start now."),
    ("ham",  "Don't worry, I've got it sorted."),
    ("ham",  "Did you hear about the new restaurant downtown?"),
    ("spam", "Last chance to win 5000 cash. Text CASH to 84484. Cost 1.50 per msg."),
]

# ─────────────────────────────────────────────
#  TEXT PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Clean and normalize SMS text."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' URL ', text)
    text = re.sub(r'\b\d[\d\s]{6,}\d\b', ' PHONE ', text)  # phone numbers
    text = re.sub(r'£|\$|€', ' CURRENCY ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
def load_dataset(filepath: str = None):
    """Load dataset from file or use built-in demo data."""
    if filepath and os.path.exists(filepath):
        print(f"📂 Loading dataset from: {filepath}")
        # Try different separators (Kaggle file uses tab)
        for sep in ['\t', ',']:
            try:
                df = pd.read_csv(filepath, sep=sep, header=None,
                                 names=['label', 'message'],
                                 encoding='latin-1', usecols=[0, 1])
                df = df.dropna()
                df['label'] = df['label'].str.strip().str.lower()
                df = df[df['label'].isin(['ham', 'spam'])]
                if len(df) > 10:
                    print(f"   ✅ Loaded {len(df):,} messages")
                    return df
            except Exception:
                continue
        print("   ⚠️  Could not parse file, using demo dataset")

    print("📋 Using built-in demo dataset (100 messages)")
    print("   💡 For full results, download from Kaggle and pass --data spam.csv")
    df = pd.DataFrame(DEMO_MESSAGES, columns=['label', 'message'])
    return df

# ─────────────────────────────────────────────
#  MODEL DEFINITIONS
# ─────────────────────────────────────────────
def get_models():
    """Return dict of model pipelines to evaluate."""
    tfidf = TfidfVectorizer(
        preprocessor=preprocess_text,
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        min_df=2
    )
    tfidf_simple = TfidfVectorizer(
        preprocessor=preprocess_text,
        ngram_range=(1, 1),
        max_features=5000,
        sublinear_tf=True
    )
    count_vec = CountVectorizer(
        preprocessor=preprocess_text,
        ngram_range=(1, 2),
        max_features=10000,
        min_df=2
    )

    models = {
        "Naive Bayes (TF-IDF)": Pipeline([
            ("tfidf", tfidf_simple),
            ("clf", MultinomialNB(alpha=0.1))
        ]),
        "Complement NB (TF-IDF)": Pipeline([
            ("tfidf", tfidf_simple),
            ("clf", ComplementNB(alpha=0.1))
        ]),
        "Logistic Regression (TF-IDF bigrams)": Pipeline([
            ("tfidf", tfidf),
            ("clf", LogisticRegression(C=5.0, max_iter=1000, solver='lbfgs'))
        ]),
        "SVM / LinearSVC (TF-IDF bigrams)": Pipeline([
            ("tfidf", tfidf),
            ("clf", LinearSVC(C=1.0, max_iter=2000))
        ]),
        "Naive Bayes (Count Vectors)": Pipeline([
            ("cv", count_vec),
            ("clf", MultinomialNB(alpha=0.5))
        ]),
    }
    return models

# ─────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Train and evaluate a single model."""
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)

    acc   = accuracy_score(y_test, y_pred)
    prec  = precision_score(y_test, y_pred, pos_label=1)
    rec   = recall_score(y_test, y_pred, pos_label=1)
    f1    = f1_score(y_test, y_pred, pos_label=1)
    cm    = confusion_matrix(y_test, y_pred)

    # Cross-val F1
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(model, X_train, y_train,
                             cv=cv, scoring='f1').mean()

    return {
        "name": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "cv_f1": cv_f1,
        "confusion_matrix": cm,
        "train_time_s": train_time,
        "model": model,
        "y_pred": y_pred
    }

# ─────────────────────────────────────────────
#  FEATURE IMPORTANCE
# ─────────────────────────────────────────────
def top_spam_features(pipeline, n=20):
    """Extract top spam-indicative features from a model."""
    try:
        vectorizer = list(pipeline.named_steps.values())[0]
        classifier = list(pipeline.named_steps.values())[-1]
        feature_names = vectorizer.get_feature_names_out()

        if hasattr(classifier, 'coef_'):
            coef = classifier.coef_
            if coef.ndim > 1:
                coef = coef[0]
            idx = np.argsort(coef)[-n:][::-1]
            return [(feature_names[i], float(coef[i])) for i in idx]
        elif hasattr(classifier, 'feature_log_prob_'):
            log_prob_diff = classifier.feature_log_prob_[1] - classifier.feature_log_prob_[0]
            idx = np.argsort(log_prob_diff)[-n:][::-1]
            return [(feature_names[i], float(log_prob_diff[i])) for i in idx]
    except Exception:
        pass
    return []

# ─────────────────────────────────────────────
#  MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────
def run_training(data_path=None):
    print("\n" + "="*60)
    print("  SMS SPAM CLASSIFIER — Training Pipeline")
    print("="*60)

    # 1. Load data
    df = load_dataset(data_path)
    df['processed'] = df['message'].apply(preprocess_text)

    le = LabelEncoder()
    y = le.fit_transform(df['label'])   # ham=0, spam=1
    X = df['message']

    spam_count = (y == 1).sum()
    ham_count  = (y == 0).sum()
    print(f"\n📊 Dataset Overview:")
    print(f"   Total messages : {len(df):,}")
    print(f"   Ham (legit)    : {ham_count:,} ({100*ham_count/len(df):.1f}%)")
    print(f"   Spam           : {spam_count:,} ({100*spam_count/len(df):.1f}%)")

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n   Train size: {len(X_train):,} | Test size: {len(X_test):,}")

    # 3. Train all models
    print("\n" + "─"*60)
    print("  TRAINING MODELS")
    print("─"*60)

    models = get_models()
    results = []
    best_result = None

    for name, model in models.items():
        print(f"\n  🔧 {name}")
        r = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(r)
        print(f"     Accuracy : {r['accuracy']:.4f}")
        print(f"     Precision: {r['precision']:.4f}  Recall: {r['recall']:.4f}")
        print(f"     F1 Score : {r['f1']:.4f}  (CV F1: {r['cv_f1']:.4f})")
        print(f"     Train time: {r['train_time_s']*1000:.1f}ms")

        if best_result is None or r['f1'] > best_result['f1']:
            best_result = r

    # 4. Comparison table
    print("\n" + "="*60)
    print("  MODEL COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Model':<42} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    print("─"*68)
    for r in sorted(results, key=lambda x: x['f1'], reverse=True):
        marker = " ★" if r['name'] == best_result['name'] else ""
        short = r['name'][:40]
        print(f"{short:<42} {r['accuracy']:>6.3f} {r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f}{marker}")

    # 5. Best model details
    print(f"\n🏆 Best Model: {best_result['name']}")
    print("\n   Confusion Matrix:")
    cm = best_result['confusion_matrix']
    print(f"   {'':12} Pred Ham  Pred Spam")
    print(f"   {'Actual Ham':<12} {cm[0,0]:>8}  {cm[0,1]:>9}")
    print(f"   {'Actual Spam':<12} {cm[1,0]:>8}  {cm[1,1]:>9}")

    tn, fp, fn, tp = cm.ravel()
    print(f"\n   True Negatives (ham→ham)  : {tn}")
    print(f"   False Positives (ham→spam): {fp}")
    print(f"   False Negatives (spam→ham): {fn}")
    print(f"   True Positives (spam→spam): {tp}")

    # 6. Top spam features
    print(f"\n📝 Top Spam Indicators (from best model):")
    features = top_spam_features(best_result['model'])
    for i, (word, score) in enumerate(features[:15], 1):
        bar = "█" * min(int(score * 5 + 1), 20)
        print(f"   {i:>2}. {word:<20} {bar}")

    # 7. Save best model
    model_path = "best_spam_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': best_result['model'],
            'label_encoder': le,
            'model_name': best_result['name']
        }, f)
    print(f"\n💾 Best model saved to: {model_path}")

    # 8. Save results JSON for the dashboard
    results_json = []
    for r in results:
        cm = r['confusion_matrix']
        results_json.append({
            "name": r['name'],
            "accuracy": round(r['accuracy'], 4),
            "precision": round(r['precision'], 4),
            "recall": round(r['recall'], 4),
            "f1": round(r['f1'], 4),
            "cv_f1": round(r['cv_f1'], 4),
            "confusion_matrix": cm.tolist(),
            "train_time_ms": round(r['train_time_s'] * 1000, 1)
        })
    with open("results.json", "w") as f:
        json.dump({
            "results": results_json,
            "best_model": best_result['name'],
            "dataset_size": len(df),
            "spam_count": int(spam_count),
            "ham_count": int(ham_count),
            "top_features": top_spam_features(best_result['model'])
        }, f, indent=2)

    return results, best_result, le

# ─────────────────────────────────────────────
#  PREDICTION MODE
# ─────────────────────────────────────────────
def predict_message(message: str, model_path: str = "best_spam_model.pkl"):
    """Predict whether a single message is spam."""
    if not os.path.exists(model_path):
        print("⚠️  No trained model found. Run training first.")
        return

    with open(model_path, 'rb') as f:
        saved = pickle.load(f)

    model = saved['model']
    le    = saved['label_encoder']

    pred = model.predict([message])[0]
    label = le.inverse_transform([pred])[0]

    print(f"\n📱 Message: \"{message}\"")
    print(f"🔍 Prediction: {'🚨 SPAM' if label == 'spam' else '✅ HAM (Legitimate)'}")

    # Try to get probability
    try:
        proba = model.predict_proba([message])[0]
        spam_prob = proba[le.transform(['spam'])[0]]
        print(f"📊 Spam probability: {spam_prob:.1%}")
    except AttributeError:
        pass  # LinearSVC doesn't support predict_proba by default

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMS Spam Classifier")
    parser.add_argument("--data", type=str, help="Path to SMS dataset CSV/TSV")
    parser.add_argument("--predict", type=str, help="Predict a single message")
    args = parser.parse_args()

    if args.predict:
        predict_message(args.predict)
    else:
        run_training(args.data)