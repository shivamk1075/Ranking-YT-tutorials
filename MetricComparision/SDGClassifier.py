# compare_models_sgd_multiclass.py

import os
os.makedirs("logs", exist_ok=True)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from tqdm import tqdm
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ---------- Setup ----------
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stops = set(stopwords.words('english'))
stops.discard('not')  # Important for sentiment

def preprocess(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stops]
    return ' '.join(tokens)

# ---------- Load Dataset ----------
DATA_PATH = "test.csv"
df = pd.read_csv(DATA_PATH)

# Keep all three classes
df = df[df['Sentiment'].str.upper().isin(["POSITIVE", "NEGATIVE", "NEUTRAL"])]

comments = df['CommentText'].astype(str).tolist()
true_labels = df['Sentiment'].str.upper().tolist()

# ---------- Load Vectorizer + Model ----------
print("\n Loading TF-IDF + SGDClassifier...")
vectorizer = joblib.load(r"C:\Users\natur\Desktop\Technologies\Projects\Ranking-YT-Tutorials\Original\SDGClassifier model\src\tfidf_vectorizer.pkl")
classifier = joblib.load(r"C:\Users\natur\Desktop\Technologies\Projects\Ranking-YT-Tutorials\Original\SDGClassifier model\src\SGDClassifier_model.pkl")

# ---------- Predict ----------
print("\n Predicting with SGDClassifier...")
processed_comments = [preprocess(c) for c in tqdm(comments)]
X_test = vectorizer.transform(processed_comments)
preds_raw = classifier.predict(X_test)

#  Map numeric predictions to labels
label_map = {
    0: "NEGATIVE",
    1: "NEUTRAL",
    2: "POSITIVE"
}
preds_sgd = [label_map[p] for p in preds_raw]

# ---------- Save Predictions ----------
df['pred_sgd'] = preds_sgd
df.to_csv("logs/sgd_predictions_multiclass.csv", index=False)

# ---------- Evaluation ----------
print("\n Evaluation for SGDClassifier (Multiclass):")
sgd_report = classification_report(true_labels, preds_sgd, digits=4)
print(sgd_report)

with open("logs/sgd_report_multiclass.txt", "w", encoding="utf-8") as f:
    f.write(" Evaluation for SGDClassifier (Multiclass):\n")
    f.write(sgd_report)

# --- Confusion Matrix ---
sgd_cm = confusion_matrix(true_labels, preds_sgd, labels=["NEGATIVE", "NEUTRAL", "POSITIVE"])
plt.figure(figsize=(6, 5))
sns.heatmap(sgd_cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=["NEGATIVE", "NEUTRAL", "POSITIVE"],
            yticklabels=["NEGATIVE", "NEUTRAL", "POSITIVE"])
plt.title("Confusion Matrix - SGDClassifier (Multiclass)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("logs/sgd_confusion_matrix_multiclass.png")
plt.close()

# --- Bar Plot for Precision, Recall, F1 ---
sgd_report_dict = classification_report(true_labels, preds_sgd, output_dict=True)
df_sgd_report = pd.DataFrame(sgd_report_dict).transpose().iloc[:3]  # NEGATIVE, NEUTRAL, POSITIVE

plt.figure(figsize=(8, 5))
df_sgd_report[["precision", "recall", "f1-score"]].plot(kind='bar', color=["#9b59b6", "#5dade2", "#58d68d"])
plt.title("Precision, Recall, F1-Score by Class - SGDClassifier (Multiclass)")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.ylabel("Score")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("logs/sgd_scores_barplot_multiclass.png")
plt.close()
