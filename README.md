# Ranking YouTube Tutorials

This project explores ranking YouTube tutorials by analyzing the sentiment of their comments using three distinct machine learning approaches. The goal is to provide a robust, reproducible pipeline for sentiment-based ranking, with modular code for data collection, processing, modeling, and visualization.

## Project Structure
- **Complex model/**: Implements a custom fine-tuned DistilBERT model (TensorFlow) for sentiment classification. Includes scripts for model conversion, fine-tuning, and inference. Uses HuggingFace Transformers and TensorFlow for deep learning.
- **Dist-Bert model/**: Uses the pre-trained `distilbert-base-uncased-finetuned-sst-2-english` model from HuggingFace for efficient, out-of-the-box sentiment analysis. All core steps (fetch, preprocess, classify, aggregate, visualize) are modularized in `src/`.
- **SDGClassifier model/**: Classical ML pipeline using TF-IDF vectorization and SGDClassifier (scikit-learn). Includes custom text preprocessing and label mapping. Model and vectorizer are serialized with joblib for fast inference.

Each approach features:
- `app.py`: Flask web server with SocketIO for real-time interaction and a YouTube video ID extraction utility.
- `src/` modules:
  - `yt_search.py`: Search YouTube for videos by keyword.
  - `data_fetch.py`: Download comments for a given video.
  - `preprocess.py`: Clean and normalize comment text.
  - `classify.py`: Predict sentiment labels for comments (using the respective model for each approach).
  - `aggregate.py`: Summarize comment-level sentiment to a video-level score.
  - `visualize.py`: Generate visualizations of sentiment results.
  - `evaluate.py`: Evaluate model performance (where applicable).

- **UI**: `client/` and `yt-sentiment-ui/` provide React-based frontends for demo and visualization.

## Key Features
- End-to-end pipeline: Search, fetch, preprocess, classify, aggregate, and visualize.
- Three modeling strategies: Custom fine-tuned transformer, pre-trained transformer, and classical ML.
- Modular, reusable code for each pipeline step.
- Jupyter notebooks for experimentation and reproducibility.
- Flask backend with real-time updates via SocketIO.

## Usage Notes
- Model weights and datasets are excluded from the repository due to size constraints. Only scripts and notebooks are provided.
- To run locally, install dependencies from `requirements.txt` in each folder and follow the scripts for your chosen approach.

## References & Inspiration
This project draws inspiration and methodology from the following research papers:

- [YouTube Comments Sentiment Analysis (Ritika Singh et al.)](https://ritikasingh95.github.io/Documents/Publications/YOUTUBE%20COMMENTS%20SENTIMENT%20ANALYSIS.pdf):
  - Motivated the use of multiple machine learning algorithms (Naïve Bayes, SVM, Logistic Regression, Decision Tree, KNN, Random Forest, and SGDClassifier) for sentiment classification of YouTube comments.
  - Emphasized the importance of preprocessing steps such as lemmatization, n-gramming, tokenization, stopword and punctuation removal, which are implemented in this project.
  - Highlighted the use of annotated corpora and evaluation metrics like F-score and accuracy, which guided the evaluation and validation steps in this work.

- [Ranking of tutorials on YouTube based on the analysis of feelings made to their comments (Goyzueta Torres et al., Innosoft Journal)](https://revistas.ulasalle.edu.pe/innosoft/article/view/66/71):
  - Inspired the core idea of ranking YouTube tutorials by aggregating sentiment scores from user comments.
  - Provided the rationale for using BERT-based models for large-scale, automated sentiment analysis and video ranking, which is reflected in the transformer-based approaches in this project.
  - Reinforced the pipeline structure: fetching comments, preprocessing, sentiment analysis, and aggregation for ranking, all of which are implemented here.
  - Stressed the value of helping viewers choose relevant content and supporting creators with actionable feedback from large-scale sentiment analysis.

These works provided both conceptual and technical foundations for the design and implementation of this project, especially in the areas of preprocessing, model selection, evaluation, and the overall ranking pipeline.

---

Thanks for exploring this project! For questions or suggestions, feel free to open an issue or reach out.
