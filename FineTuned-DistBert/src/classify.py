from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TextClassificationPipeline

MODEL_NAME = r"C:\Users\natur\Desktop\Projects\Ranking-YT-Tutorials\Complex model\src\distilbert-finetuned-youtube-tf"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

sentiment_pipeline = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    return_all_scores=False,
    framework="tf",
    device=0
)


def classify_comments(comments: list[str]) -> list[str]:
    """
    Given a list of cleaned comment strings, return a list of sentiment labels.
    """
    print(f"Classifying {len(comments)} comments...")
    results = sentiment_pipeline(comments, batch_size=16)
    print(f"Results: {results[:3]}")
    return [res["label"] for res in results]

if __name__ == "__main__":
    samples = [
        "this video is fantastic and very helpful",
        "i did not find this useful at all"
    ]
    labels = classify_comments(samples)
    print(list(zip(samples, labels)))
