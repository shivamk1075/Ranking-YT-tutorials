import os
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv

# your pipeline modules
from src.yt_search import yt_search
from src.data_fetch import fetch_comments
from src.preprocess import preprocess_comments
from src.classify import classify_comments
from src.aggregate import aggregate_video_sentiment

load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET', 'secret!')
socketio = SocketIO(app, cors_allowed_origins="*")

import re
from urllib.parse import urlparse, parse_qs


import nltk
nltk.data.path.append('nltk_data')

def extract_video_id(url_or_id: str) -> str | None:
    # Case 1: Already a valid 11-character ID (e.g., "p1bfK8ZJgkE")
    if re.fullmatch(r"[\w-]{11}", url_or_id):
        return url_or_id

    # Case 2: youtu.be short links (e.g., https://youtu.be/p1bfK8ZJgkE)
    parsed = urlparse(url_or_id)
    if parsed.netloc in ("youtu.be", "www.youtu.be"):
        vid = parsed.path.lstrip("/").split("/")[0]
        return vid if re.fullmatch(r"[\w-]{11}", vid) else None

    # Case 3: Standard YouTube watch URLs (e.g., https://www.youtube.com/watch?v=...)
    if "youtube.com" in parsed.netloc:
        if parsed.path == "/watch":
            vid = parse_qs(parsed.query).get("v", [None])[0]
            return vid if vid and re.fullmatch(r"[\w-]{11}", vid) else None
        # Handle /embed/ID, /v/ID, /shorts/ID
        path_parts = parsed.path.split("/")
        if len(path_parts) >= 3 and path_parts[1] in ("embed", "v", "shorts"):
            vid = path_parts[2]
            return vid if re.fullmatch(r"[\w-]{11}", vid) else None

    # Case 4: Fallback regex for non-standard URLs (e.g., "https://m.youtube.com/watch?v=...")
    match = re.search(
        r"(?:v=|v/|vi=|vi/|youtu\.be/|/v/|/e/|embed/|shorts/)([\w-]{11})", 
        url_or_id
    )
    return match.group(1) if match else None



@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_pipeline')
def handle_start(data):
    raw_input = data.get('video_url', '').strip()
    print(f"Received input: '{raw_input}'")  # <-- Add this
    video_id = extract_video_id(raw_input)
    print(f"Extracted video_id: '{video_id}'")  # <-- And this
    if not video_id:
        emit('log', {'msg': '❌ Invalid YouTube video URL or ID.'})
        # emit('log', {'msg':video_id})
        return  # Stop if invalid
    # video_id = data.get('video_id')
    max_comments = int(data.get('max_comments', 100))
    emit('log', {'msg': f'Fetching up to {max_comments} comments for {video_id}…'})
    raw = fetch_comments(video_id, max_results=max_comments)
    emit('log', {'msg': f'Fetched {len(raw)} comments. Cleaning…'})
    cleaned = preprocess_comments(raw)
    emit('log', {'msg': 'Comments cleaned. Classifying…'})
    labels = classify_comments(cleaned)
    emit('log', {'msg': 'Classification done. Aggregating…'})
    video_label = aggregate_video_sentiment(cleaned, labels)
    emit('log', {'msg': f'Video-level label: {video_label}'})
    # send final result
    emit('result', {
        'video_label': video_label,
        'distribution': {
            'POSITIVE': labels.count('POSITIVE'),
            # 'NEUTRAL':  labels.count('NEUTRAL'),
            'NEGATIVE': labels.count('NEGATIVE')
        }
    })

@socketio.on('search_topic')
def handle_search_topic(data):
    search_term = data.get('search_term', '').strip()
    limit = int(data.get('limit', 5))
    emit('log', {'msg': f"Searching YouTube for '{search_term}'..."})

    try:
        search_results = yt_search(search_term, limit=limit)
    except Exception as e:
        emit('log', {'msg': f"Search failed: {str(e)}"})
        emit('search_results', {'results': []})
        return

    recommendations = []

    for result in search_results:
        emit('log', {'msg': f"Analyzing: {result.title} ({result.url})"})
        try:
            comments = fetch_comments(result.id, max_results=100)
            cleaned = preprocess_comments(comments)
            labels = classify_comments(cleaned)
            video_label = aggregate_video_sentiment(cleaned, labels)
            recommendations.append({
                "title": result.title,
                "uploader": result.uploader,
                "url": result.url,
                "duration": result.duration,
                "video_id": result.id,
                "video_label": video_label,
                "positive": labels.count("POSITIVE"),
                # "neutral": labels.count("NEUTRAL"),
                "negative": labels.count("NEGATIVE")
            })
        except Exception as e:
            emit('log', {'msg': f"Skipping {result.title}: {str(e)}"})

    # Sort: Useful > Partially Useful > Not Useful
    useful = [r for r in recommendations if r["video_label"] == "Useful"]
    partially = [r for r in recommendations if r["video_label"] == "Partially Useful"]
    not_useful = [r for r in recommendations if r["video_label"] == "Not Useful"]
    sorted_results = useful + partially + not_useful

    emit('search_results', {'results': sorted_results})


if __name__ == '__main__':
        # Test Cases
    print(extract_video_id("p1bfK8ZJgkE"))  # Direct ID → "p1bfK8ZJgkE"
    print(extract_video_id("https://www.youtube.com/watch?v=p1bfK8ZJgkE&t=784s"))  # Standard URL → "p1bfK8ZJgkE"
    print(extract_video_id("https://youtu.be/p1bfK8ZJgkE"))  # Short link → "p1bfK8ZJgkE"
    print(extract_video_id("https://www.youtube.com/embed/p1bfK8ZJgkE"))  # Embed → "p1bfK8ZJgkE"
    print(extract_video_id("https://www.youtube.com/shorts/p1bfK8ZJgkE"))  # Shorts → "p1bfK8ZJgkE"
    print(extract_video_id("invalid"))  # → None

    # import eventlet
    # eventlet.monkey_patch()

    # if __name__ == '__main__':
    #     socketio.run(app, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
    import eventlet
    eventlet.monkey_patch()

    if __name__ == '__main__':
        socketio.run(app, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
    # socketio.run(app, host='0.0.0.0', port=5000, debug=True)


