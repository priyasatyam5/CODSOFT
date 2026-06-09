# 🤖 Machine Learning Projects — Internship Portfolio

A collection of 3 end-to-end Machine Learning projects built during my internship, covering real-world problems in text classification, financial fraud detection, and customer behaviour prediction.

📁 Projects Overview

| Project | Type | Best Model | Accuracy |
|---|---|---|---|
| 📩 SMS Spam Classifier | Text Classification | SVM + TF-IDF | 98.6% |
| 💳 Credit Card Fraud Detection | Binary Classification | Random Forest | ~99.0% |
| 📉 Customer Churn Prediction | Binary Classification | Gradient Boosting | ~85–90% |

📩 1. SMS Spam Classifier

Classifies SMS messages as **spam** or **ham (legitimate)** using NLP techniques and multiple classifiers.

Dataset
- **Source:** [UCI SMS Spam Collection — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size:** 5,572 messages (4,825 ham · 747 spam)

Techniques Used
- **Text preprocessing** — lowercasing, URL/phone number tokenization
- **Feature extraction** — TF-IDF (unigrams + bigrams), Count Vectorizer
- **Models trained** — Naive Bayes, Complement NB, Logistic Regression, SVM (LinearSVC)
- **Evaluation** — Accuracy, Precision, Recall, F1, 5-fold Cross Validation, Confusion Matrix

Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| SVM / LinearSVC ⭐ | 98.65% | 98.55% | 91.28% | 94.77% |
| Logistic Regression | 98.39% | 99.25% | 88.59% | 93.62% |
| Complement NB | 98.30% | 92.21% | 95.30% | 93.73% |
| Naive Bayes (TF-IDF) | 98.21% | 97.78% | 88.59% | 92.96% |
| Naive Bayes (Count) | 98.39% | 96.45% | 91.28% | 93.79% |

Confusion Matrix (Best Model — SVM)
```
                Predicted Ham   Predicted Spam
Actual Ham           964              2
Actual Spam           13            136
```
- ✅ Only **2 legitimate messages** wrongly flagged as spam
- ✅ Only **13 spam messages** slipped through

Top Spam Keywords Detected
`txt` · `call` · `uk` · `text` · `reply` · `150p` · `mobile` · `ringtone` · `claim` · `free` · `won`

How to Run
```bash
# Install dependencies
pip install scikit-learn pandas numpy

# Train all models
python sms_spam_classifier.py --data spam.csv

# Predict a single message
python sms_spam_classifier.py --predict "Congratulations! You won a free prize, claim now!"
```


💳 2. Credit Card Fraud Detection

Detects **fraudulent credit card transactions** from a highly imbalanced real-world dataset.

Dataset
- **Source:** [Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions (only 0.17% are fraud)

Techniques Used
- **Imbalanced data handling** — SMOTE (Synthetic Minority Oversampling), class weighting
- **Feature scaling** — StandardScaler on `Amount` and `Time`
- **Models trained** — Logistic Regression, Random Forest, XGBoost
- **Evaluation** — Precision, Recall, F1, ROC-AUC, Confusion Matrix

Key Challenge
The dataset is extremely imbalanced — **99.83% legitimate vs 0.17% fraud**. A model that predicts everything as legitimate would have 99.83% accuracy but catch zero fraud. This is why **Recall** and **ROC-AUC** are the important metrics here.

How to Run
```bash
pip install scikit-learn pandas numpy imbalanced-learn xgboost

python fraud_detection.py --data creditcard.csv
```

---

📉 3. Customer Churn Prediction
Predicts whether a customer is likely to **cancel their subscription or leave** a service.

Dataset
- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** ~7,000 customer records with 20 features

Techniques Used
- **Feature engineering** — encoding categorical variables, handling missing values
- **Models trained** — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
- **Evaluation** — Accuracy, F1, ROC-AUC, Feature Importance plot

Key Features That Predict Churn
- Contract type (month-to-month customers churn more)
- Tenure (newer customers churn more)
- Monthly charges (higher bills = higher churn)
- Tech support / Online security subscriptions

How to Run
```bash
pip install scikit-learn pandas numpy matplotlib seaborn

python churn_prediction.py --data WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange?logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-1.3+-lightgrey?logo=pandas)
![numpy](https://img.shields.io/badge/numpy-1.21+-blue?logo=numpy)

- **Language:** Python 3.8+
- **ML Library:** scikit-learn
- **Data handling:** pandas, numpy
- **Visualisation:** matplotlib, seaborn
- **Imbalanced learning:** imbalanced-learn (SMOTE)

📂 Folder Structure
ml-internship-projects/
│
├── sms_spam_classifier/
│   ├── sms_spam_classifier.py
│   ├── spam.csv
│   └── best_spam_model.pkl
│
├── fraud_detection/
│   ├── fraud_detection.py
│   └── creditcard.csv
│
├── customer_churn/
│   ├── churn_prediction.py
│   └── telco_churn.csv
│
└── README.md

🚀 Getting Started

1. Clone this repository
```bash
git clone https://github.com/your-username/ml-internship-projects.git
cd ml-internship-projects
```

2. Install all dependencies
```bash
pip install -r requirements.txt
```

3. Download datasets from the Kaggle links above and place them in the correct folders

4. Run any project using the commands listed above

📚 What I Learned

- Building complete ML pipelines from raw data to saved model
- Handling real-world problems like **class imbalance** and **noisy text data**
- Choosing the right metric for the right problem (F1 vs Accuracy vs ROC-AUC)
- How **TF-IDF** converts text into numbers a model can understand
- Why **SVM** works so well for text classification problems

📄 License

This project is open source and available under the [MIT License](LICENSE).

## Author
SHANMUKHA PRIYA GANTYADA

