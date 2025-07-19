# evaluate_distilbert_youtube.py

import os
os.makedirs("logs", exist_ok=True)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TextClassificationPipeline
import tensorflow as tf

# ---------- Load Dataset ----------
DATA_PATH = "test.csv"  # Update path if needed
df = pd.read_csv(DATA_PATH)

# Keep only POSITIVE and NEGATIVE rows
df = df[df['Sentiment'].str.upper().isin(["POSITIVE", "NEGATIVE"])]
comments = df['CommentText'].astype(str).tolist()
true_labels = df['Sentiment'].str.upper().tolist()

# ---------- Load Fine-tuned DistilBERT ----------
# MODEL_PATH = r"C:\Users\natur\Desktop\Projects\Ranking-YT-Tutorials\Complex model\src\distilbert-finetuned-youtube-tf"
MODEL_PATH = "./Complex model/src/distilbert-finetuned-youtube-tf"


device = 0 if tf.config.list_physical_devices('GPU') else -1
print(f"\n✅ Loading YouTube fine-tuned DistilBERT (device: {'GPU' if device == 0 else 'CPU'})...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

pipe = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    return_all_scores=False,
    framework="tf",
    device=device
)

# ---------- Predict ----------
print("\n✅ Predicting with fine-tuned DistilBERT...")
yt_preds_raw = list(tqdm(pipe(comments, batch_size=16, truncation=True), total=len(comments)))
yt_preds = [res['label'].upper() for res in yt_preds_raw]

# Save predictions
df['pred_ytbert'] = yt_preds
df.to_csv("logs/youtube_bert_predictions_binary.csv", index=False)

# ---------- Evaluation ----------
print("\n📊 Evaluation for Fine-tuned DistilBERT (Binary):")
yt_report = classification_report(true_labels, yt_preds, digits=4)
print(yt_report)

with open("logs/youtube_bert_report_binary.txt", "w", encoding="utf-8") as f:
    f.write("📊 Evaluation for Fine-tuned DistilBERT (Binary):\n")
    f.write(yt_report)

# --- Confusion Matrix ---
yt_cm = confusion_matrix(true_labels, yt_preds, labels=["NEGATIVE", "POSITIVE"])
plt.figure(figsize=(5, 4))
sns.heatmap(yt_cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=["NEGATIVE", "POSITIVE"],
            yticklabels=["NEGATIVE", "POSITIVE"])
plt.title("Confusion Matrix - Fine-tuned DistilBERT (Binary)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("logs/youtube_bert_confusion_matrix_binary.png")
plt.close()

# --- Bar Plot for Precision, Recall, F1 ---
yt_report_dict = classification_report(true_labels, yt_preds, output_dict=True)
df_yt_report = pd.DataFrame(yt_report_dict).transpose().iloc[:2]  # NEGATIVE and POSITIVE

plt.figure(figsize=(8, 5))
df_yt_report[["precision", "recall", "f1-score"]].plot(kind='bar', color=["#f39c12", "#27ae60", "#2980b9"])
plt.title("Precision, Recall, F1-Score by Class - Fine-tuned DistilBERT (Binary)")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.ylabel("Score")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("logs/youtube_bert_scores_barplot_binary.png")
plt.close()
