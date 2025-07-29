import pandas as pd
import re
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Custom stopwords list
stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
             'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself',
             'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them',
             'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
             'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
             'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
             'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
             'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
             'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
             'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
             'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
             'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
             'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
             'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'}

# Text preprocessing
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    return ' '.join([stemmer.stem(word) for word in text.split() if word not in stopwords])

df['cleaned'] = df['message'].apply(clean_text)

# Feature Extraction
cv = CountVectorizer()
X = cv.fit_transform(df['cleaned'])
y = df['label']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# --- Predict on user input ---
print("\n🔍 Enter an email message to check if it's spam or not:")
user_input = input("👉 Your message: ")

# Clean and transform the input
cleaned_input = clean_text(user_input)
vectorized_input = cv.transform([cleaned_input])

# Predict
prediction = model.predict(vectorized_input)[0]

# Output result
print("\n📢 Prediction:", "Spam 🚫" if prediction == 1 else "Not Spam ✅")
