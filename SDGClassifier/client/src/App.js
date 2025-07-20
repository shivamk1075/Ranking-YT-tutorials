import React, { useState, useEffect, useRef } from "react";
import { io } from "socket.io-client";
import "./App.css";

// const socket = io("http://localhost:5000"); // Adjust if your backend runs elsewhere
// const socket = io("https://nodistractyoutube.onrender.com", {
//   transports: ["websocket"],
// });
// const socket = io("https://rankingyt-web.onrender.com", {
//   transports: ["websocket"],
// });
const socket = io("https://rankerbackend.onrender.com/", {
  transports: ["websocket"],
});



function App() {
  const [videoId, setVideoId] = useState("");
  const [maxComments, setMaxComments] = useState(100);
  const [log, setLog] = useState([]);
  const [topic, setTopic] = useState("");
  const [results, setResults] = useState([]);
  const logRef = useRef(null);

  useEffect(() => {
  // Named handler functions
  const handleLog = (data) => setLog(prev => [...prev, data.msg]);
  const handleResult = (data) => {
    setLog(prev => [
      ...prev,
      <div key={Date.now()}>
        <b>Video-level label:</b> {data.video_label}
        <br />
        Distribution: 👍 {data.distribution.POSITIVE} | 👎 {data.distribution.NEGATIVE}
      </div>
    ]);
  };
  const handleSearchResults = (data) => setResults(data.results);

  // Add listeners
  socket.on("log", handleLog);
  socket.on("result", handleResult);
  socket.on("search_results", handleSearchResults);

  // Cleanup function
  return () => {
    socket.off("log", handleLog);
    socket.off("result", handleResult);
    socket.off("search_results", handleSearchResults);
  };
}, []); // Empty dependency array ensures this runs once


  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [log]);

  const handleRun = () => {
    setLog([]);
    setResults([]);
    if (!videoId) return alert("Enter a video ID or link");
    socket.emit("start_pipeline", { video_url: videoId, max_comments: maxComments });
  };

  const handleSearch = () => {
    setLog([]);
    setResults([]);
    if (!topic) return alert("Enter a topic");
    socket.emit("search_topic", { search_term: topic, limit: 5 });
  };

  return (
    <div className="container">
      <header>
        <i className="fas fa-graduation-cap"></i> YouTube Tutorial Sentiment
      </header>
      <div className="input-row">
        <input
          value={videoId}
          onChange={e => setVideoId(e.target.value)}
          placeholder="Paste YouTube video URL or ID..."
        />
        <input
          type="number"
          value={maxComments}
          onChange={e => setMaxComments(e.target.value)}
          min={1}
          max={1000}
          style={{ maxWidth: 120 }}
        />
        <button onClick={handleRun}><i className="fas fa-play"></i> Run</button>
      </div>
      <div className="log-box" ref={logRef}>
        {log.map((msg, i) => <div key={i}>{msg}</div>)}
      </div>
      <div className="input-row">
        <input
          value={topic}
          onChange={e => setTopic(e.target.value)}
          placeholder="Search for tutorials..."
        />
        <button onClick={handleSearch}><i className="fas fa-search"></i> Search</button>
      </div>
      <div id="searchResults">
        {results.length > 0 && (
          <>
            <h2>Recommended Tutorials:</h2>
            <ol className="tutorial-list">
              {results.map((r, idx) => (
                <li key={r.url}>
                  <a href={r.url} target="_blank" rel="noopener noreferrer" className="tutorial-title">
                    {idx + 1}. {r.title} <i className="fas fa-fire" style={{ color: "#ffb347" }}></i>
                  </a>
                  <div className="tutorial-meta">
                    by <b>{r.uploader}</b> ({r.duration})
                  </div>
                  <div>
                    <span className={
                      r.video_label === "Useful"
                        ? "sentiment-label sentiment-useful"
                        : r.video_label === "Partially Useful"
                        ? "sentiment-label sentiment-partial"
                        : "sentiment-label sentiment-not"
                    }>
                      Label: {r.video_label}
                    </span>
                    <span className="sentiment-icons">
                      <span title="Positive"><i className="fas fa-thumbs-up"></i> {r.positive}</span>
                      <span title="Neutral" style={{ marginLeft: "0.7em" }}><i className="fas fa-meh"></i> {r.neutral}</span>
                      <span title="Negative" style={{ marginLeft: "0.7em" }}><i className="fas fa-thumbs-down"></i> {r.negative}</span>
                    </span>
                  </div>
                </li>
              ))}
            </ol>
          </>
        )}
      </div>
      <footer>
        &copy; 2025 YouTube Sentiment Analyzer &mdash; React UI
      </footer>
    </div>
  );
}

export default App;
