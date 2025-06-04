# src/classify.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

# 1. Choose your model: for English you might use 'distilbert-base-uncased-finetuned-sst-2-english'
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

# 2. Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# 3. Create a pipeline for ease of use
sentiment_pipeline = TextClassificationPipeline(
    model=model,
    tokenizer=tokenizer,
    return_all_scores=False,    # get only the top label
    device=0                    # set to -1 for CPU, or GPU index if available
)

def classify_comments(comments: list[str]) -> list[str]:
    """
    Given a list of cleaned comment strings, return a list of sentiment labels.
    """
    # Pipeline expects a list of texts
    results = sentiment_pipeline(comments, batch_size=16)
    # Each result is {'label': 'POSITIVE', 'score': 0.99}
    return [res["label"] for res in results]

if __name__ == "__main__":
    # Quick check
    samples = [
        "this video is fantastic and very helpful",
        "i did not find this useful at all"
    ]
    labels = classify_comments(samples)
    print(list(zip(samples, labels)))
