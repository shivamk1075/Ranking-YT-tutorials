# main.py

from src.data_fetch import fetch_comments
from src.preprocess import preprocess_comments
from src.classify import classify_comments
from src.aggregate import aggregate_video_sentiment
from src.visualize import plot_distribution, generate_wordcloud

def main(video_id, max_comments=200):
    raw      = fetch_comments(video_id, max_results=max_comments)
    cleaned  = preprocess_comments(raw)
    labels   = classify_comments(cleaned)
    # After labels = classify_comments(...)
    plot_distribution(labels)

    # For each class
    for sentiment in ["POSITIVE","NEUTRAL","NEGATIVE"]:
        subset = [c for c,l in zip(cleaned, labels) if l==sentiment]
        if subset:
            generate_wordcloud(subset, sentiment)
    video_label = aggregate_video_sentiment(cleaned, labels)

    print(f"Video-level label: {video_label}\n")
    # Optionally, show distribution
    from collections import Counter
    print("Comment distribution:", Counter(labels))

if __name__ == "__main__":
    main("SmZmBKc7Lrs", max_comments=100)
