# 📧 Email Spam Detection with Machine Learning

This project is a **Spam Email Detector** built using **Python** and **Machine Learning (Naive Bayes)**. It classifies email messages as **spam** or **not spam** (ham) using text processing and natural language techniques.

---

## 📂 Dataset
- **File**: `spam.csv`
- **Source**: [Available publicly]
- Contains over 5,000 messages labeled as `ham` (not spam) or `spam`
- Columns used:
  - `v1`: Label (ham/spam)
  - `v2`: Message content

---

## 🛠️ Technologies Used
- Python 3
- pandas
- scikit-learn
- nltk (PorterStemmer)

---

## 📊 Model Details
 Model Used: Multinomial Naive Bayes

 Vectorizer: CountVectorizer (Bag of Words)

 Accuracy: ~97%

 markdown

---

## Classification Report:
        precision    recall  f1-score   support

           0       0.99      0.98      0.99       965
           1       0.88      0.95      0.91       150

    accuracy                           0.97      1115
   macro avg       0.93      0.96      0.95      1115
weighted avg       0.98      0.97      0.98      1115

---

