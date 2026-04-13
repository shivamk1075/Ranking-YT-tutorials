from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

sentiment_pipeline = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    return_all_scores=False,
    device=0
)

def classify_comments(comments: list[str]) -> list[str]:
    """
    Given a list of cleaned comment strings, return a list of sentiment labels.
    """
    results = sentiment_pipeline(comments, batch_size=16)
    return [res["label"] for res in results]

if __name__ == "__main__":
    samples = [
        "this video is fantastic and very helpful",
        "i did not find this useful at all"
    ]
    labels = classify_comments(samples)
    print(list(zip(samples, labels)))
