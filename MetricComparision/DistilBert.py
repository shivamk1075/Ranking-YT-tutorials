import os
os.makedirs("logs", exist_ok=True)

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import pipeline
import torch

# ---------- Load Dataset ----------
DATA_PATH = "test.csv"  # Update this path if needed!
df = pd.read_csv(DATA_PATH)

# Keep only POSITIVE and NEGATIVE rows
df = df[df['Sentiment'].str.upper().isin(["POSITIVE", "NEGATIVE"])]

comments = df['CommentText'].astype(str).tolist()
true_labels = df['Sentiment'].str.upper().tolist()

# ---------- Load DistilBERT ----------
device = 0 if torch.cuda.is_available() else -1
print(f"\nLoading DistilBERT pipeline (device: {'GPU' if device == 0 else 'CPU'})...")

pipe = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)

# ---------- Predict with DistilBERT ----------
print("\n Predicting with DistilBERT...")
bert_preds_raw = list(tqdm(pipe(comments, batch_size=32, truncation=True), total=len(comments)))
bert_preds = [res['label'].upper() for res in bert_preds_raw]

# Save predictions
df['pred_bert'] = bert_preds
df.to_csv("logs/bert_predictions_binary.csv", index=False)

# ---------- Evaluation ----------
print("\n Evaluation for DistilBERT (Binary):")
bert_report = classification_report(true_labels, bert_preds, digits=4)
print(bert_report)

with open("logs/bert_report_binary.txt", "w", encoding="utf-8") as f:
    f.write(" Evaluation for DistilBERT (Binary):\n")
    f.write(bert_report)

# --- Confusion Matrix ---
bert_cm = confusion_matrix(true_labels, bert_preds, labels=["NEGATIVE", "POSITIVE"])
plt.figure(figsize=(5, 4))
sns.heatmap(bert_cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=["NEGATIVE", "POSITIVE"],
            yticklabels=["NEGATIVE", "POSITIVE"])
plt.title("Confusion Matrix - DistilBERT (Binary)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("logs/bert_confusion_matrix_binary.png")
plt.close()

# --- Bar Plot for Precision, Recall, F1 ---
bert_report_dict = classification_report(true_labels, bert_preds, output_dict=True)
df_bert_report = pd.DataFrame(bert_report_dict).transpose().iloc[:2]  # NEGATIVE and POSITIVE

plt.figure(figsize=(8, 5))
df_bert_report[["precision", "recall", "f1-score"]].plot(kind='bar', color=["#9b59b6", "#5dade2", "#58d68d"])
plt.title("Precision, Recall, F1-Score by Class - DistilBERT (Binary)")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.ylabel("Score")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("logs/bert_scores_barplot_binary.png")
plt.close()
