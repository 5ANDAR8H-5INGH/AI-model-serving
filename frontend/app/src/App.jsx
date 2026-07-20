import { useState } from "react";
import "./App.css";

function App() {
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const callAPI = async (endpoint) => {
    if (!prompt.trim()) return;

    setLoading(true);
    setResponse("");

    try {
      const res = await fetch(`http://127.0.0.1:8000/${endpoint}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: prompt,
        }),
      });

      const data = await res.json();

if (endpoint === "qna") {
  setResponse(data.Response);
}
else if (endpoint === "generate_text") {
  setResponse(data.Response);
}
else if (endpoint === "summarize") {
  setResponse(data.Summary);
}
else if (endpoint === "sentiment_analysis") {
  setResponse(
    `Sentiment: ${data.Sentiment}\nConfidence: ${data.Score}%`
  );
}
    } catch (error) {
      setResponse("Error connecting to backend");
    }

    setLoading(false);
  };

  return (
    <div className="container">
      <div className="card">
        <h1>AI Toolkit</h1>
        <p className="subtitle">
         
        </p>

        <textarea
          placeholder="Enter your text here..."
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />

        <div className="button-group">
          <button onClick={() => callAPI("generate_text")}>
            Generate
          </button>

          <button onClick={() => callAPI("summarize")}>
            Summarize
          </button>

          <button onClick={() => callAPI("sentiment_analysis")}>
            Sentiment
          </button>

          <button onClick={() => callAPI("qna")}>
            Q&A
          </button>
        </div>

        <div className="output">
          {loading ? (
            <p>Processing...</p>
          ) : (
            <pre>{response}</pre>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;