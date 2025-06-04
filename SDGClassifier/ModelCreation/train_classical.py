# train_classical.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib
import re
from tqdm import tqdm  # Add this import

# 1. Load your dataset
df = pd.read_csv("youtube-comments-sentiment.csv")
df = df.sample(frac=0.3, random_state=42).reset_index(drop=True)
comments = df["CommentText"].tolist()
labels = df["Sentiment"].map({"Negative":0, "Neutral":1, "Positive":2}).tolist()

# 2. Preprocess (same as your BERT preprocessing)
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

comments_clean = [preprocess(c) for c in comments]

# 3. Vectorize with TF-IDF (better than CountVectorizer)
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
X = vectorizer.fit_transform(comments_clean)
y = labels

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 5. Define classifiers
models = {
    "Naive Bayes": GaussianNB(),
    "SVM": SGDClassifier(loss='hinge', max_iter=1000),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# 6. Train & Evaluate
results = {}
# ...existing code...
for name, model in tqdm(models.items(), desc="Training models"):  # Wrap with tqdm
    print(f"\nTraining {name}...")
    if name == "Naive Bayes":
        model.fit(X_train.toarray(), y_train)
        y_pred = model.predict(X_test.toarray())
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    # ...existing code...
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)
    
    results[name] = {
        "accuracy": round(acc, 3),
        "f1_score": round(f1, 3),
        "confusion_matrix": cm
    }
    
# 7. Save models and vectorizer
joblib.dump(vectorizer, "classical_models/tfidf_vectorizer.pkl")
for name, model in models.items():
    joblib.dump(model, f"classical_models/{name.replace(' ', '_').lower()}_model.pkl")

# 8. Print results
print("\n=== Results ===")
for model, metrics in results.items():
    print(f"{model}:")
    print(f"  Accuracy: {metrics['accuracy']}")
    print(f"  Macro-F1: {metrics['f1_score']}")
    print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}\n")
