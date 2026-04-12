<img src="data/Sentiment_analysis.png" alt="Header image showing sentiment analysis visualization" width="100%"/>

# YouTube Tutorial Ranker

### Using NLP and sentiment analysis to rank YouTube tutorials by the quality of their audience reception

_A modular, end-to-end pipeline combining transformer models and classical ML to surface the best tutorials on any topic. By [Your Name]_

Finding a **good tutorial on YouTube** is harder than it should be. View counts and likes are gameable, thumbnails are misleading, and the algorithm optimizes for engagement — not educational quality. But **comments don't lie**. Viewers who genuinely learn something say so. Viewers who waste 20 minutes say that too.

This project treats **comment sentiment as a proxy for tutorial quality** and builds a full pipeline — from search to ranked output — that lets you compare tutorials on any topic through the lens of how their audiences actually responded.

### Goal of the Project

The goal was to build a **reproducible, modular ranking system** that:
1. Searches YouTube for tutorials on a given keyword
2. Fetches and preprocesses comments at scale
3. Classifies sentiment using one of three ML approaches
4. Aggregates comment-level scores into a **per-video ranking**

Three independent modeling strategies were implemented and compared — from a classical TF-IDF pipeline to a custom fine-tuned transformer.

### What I Did

- I implemented **three distinct sentiment classification approaches**, each fully self-contained:

  - **Custom Fine-Tuned DistilBERT** (`Complex model/`): A DistilBERT model fine-tuned end-to-end on labeled YouTube comment data using TensorFlow and HuggingFace Transformers. Includes scripts for fine-tuning, model conversion, and inference.
  - **Pre-Trained DistilBERT** (`Dist-Bert model/`): Uses `distilbert-base-uncased-finetuned-sst-2-english` off the shelf — no training required. Fast and strong baseline for comparison.
  - **SGDClassifier** (`SDGClassifier model/`): A classical ML pipeline using TF-IDF vectorization and a Stochastic Gradient Descent classifier (scikit-learn). Model and vectorizer serialized with `joblib` for lightweight deployment.

- Each approach follows the **same modular pipeline** structured under `src/`:
  - `yt_search.py` — Search YouTube for videos by keyword
  - `data_fetch.py` — Download comments for a given video ID
  - `preprocess.py` — Clean and normalize comment text (tokenization, stopword removal, lemmatization)
  - `classify.py` — Predict sentiment labels using the approach-specific model
  - `aggregate.py` — Summarize comment-level predictions into a video-level score
  - `visualize.py` — Generate visualizations of per-video sentiment distributions
  - `evaluate.py` — Evaluate model performance with F-score and accuracy metrics

- A **Flask backend with SocketIO** powers real-time updates as comments are fetched and classified, exposed through a **React frontend** for live demo and visualization.

- **Jupyter notebooks** are included throughout for experimentation, model evaluation, and reproducibility.

### Architecture

```
Keyword Input
     │
     ▼
YouTube Search (yt_search.py)
     │
     ▼
Comment Fetch (data_fetch.py)
     │
     ▼
Preprocessing (preprocess.py)
     │
     ▼
Sentiment Classification ──────┬─── Custom DistilBERT (fine-tuned)
     │                         ├─── Pre-trained DistilBERT (SST-2)
     │                         └─── TF-IDF + SGDClassifier
     ▼
Score Aggregation (aggregate.py)
     │
     ▼
Ranked Tutorial List + Visualizations
```

### Use

Each modeling approach is self-contained in its own folder with its own `requirements.txt`. To get started:

```bash
git clone https://github.com/your-username/youtube-tutorial-ranker.git
cd youtube-tutorial-ranker/
```

Choose your preferred approach and install its dependencies:

```bash
# Example: Pre-trained DistilBERT approach
cd "Dist-Bert model/"
pip install -r requirements.txt
```

Then launch the Flask backend:

```bash
python app.py
```

And in a separate terminal, start the React frontend:

```bash
cd yt-sentiment-ui/
npm install
npm start
```

> **Note:** Model weights and datasets are excluded from the repository due to size constraints. Scripts and notebooks for training are included — follow the fine-tuning notebooks to reproduce the weights for the custom DistilBERT approach.

### Examples

_[Add screenshot of the React frontend with ranked tutorial list]_

#### Sentiment Distribution per Video

_[Add example visualization: bar chart or heatmap of sentiment scores across videos]_

#### Ranking Output

_[Add example table: video titles, sentiment scores, and final rank]_

### Model Comparison

| Approach | Type | Training Required | Speed | Notes |
|---|---|---|---|---|
| Custom DistilBERT | Transformer (fine-tuned) | Yes | Slow | Highest domain fit |
| Pre-trained DistilBERT | Transformer (SST-2) | No | Medium | Strong zero-shot baseline |
| TF-IDF + SGDClassifier | Classical ML | Yes (fast) | Fast | Lightweight deployment |

### References & Inspiration

- [**YouTube Comments Sentiment Analysis** — Ritika Singh et al.](https://ritikasingh95.github.io/Documents/Publications/YOUTUBE%20COMMENTS%20SENTIMENT%20ANALYSIS.pdf): Motivated the multi-algorithm comparison and the preprocessing pipeline (lemmatization, tokenization, stopword removal, n-grams). Guided evaluation using F-score and accuracy.

- [**Ranking of tutorials on YouTube based on the analysis of feelings made to their comments** — Goyzueta Torres et al., Innosoft Journal](https://revistas.ulasalle.edu.pe/innosoft/article/view/66/71): Directly inspired the core idea of using aggregated comment sentiment as a ranking signal, and the rationale for BERT-based approaches at scale.

### Thanks

- ... to the **HuggingFace team** for the Transformers library and the pre-trained SST-2 DistilBERT model.
- ... to **Ritika Singh et al.** and **Goyzueta Torres et al.** for the research that shaped this project's design.
- ... to the open-source community behind `scikit-learn`, `Flask`, and `SocketIO` for making rapid ML prototyping accessible.

---

_For questions or suggestions, feel free to open an issue or reach out._
