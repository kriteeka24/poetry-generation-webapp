import React, { useState } from "react";
import { generatePoem } from "./api";

type ModelName = "gpt2" | "lstm";

export default function Generator() {
  const [model, setModel] = useState<ModelName>("gpt2");
  const [prompt, setPrompt] = useState("A Quiet Morning"); // default prompt/title
  const [maxLength, setMaxLength] = useState(120);
  const [temperature, setTemperature] = useState(0.8);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const samplePrompts = [
    "A Quiet Morning",
    "Lonely River",
    "The Last Sunset",
    "Whispers of the Old Oak",
  ];

  async function onGenerate() {
    setError(null);
    setResult("");
    setLoading(true);
    try {
      const res = await generatePoem({
        model,
        prompt,
        max_length: maxLength,
        temperature,
      });
      setResult(res.generated_text ?? "");
    } catch (err: any) {
      console.error(err);
      setError(err?.response?.data?.error ?? err?.message ?? "Unknown error");
    } finally {
      setLoading(false);
    }
  }

  function downloadPoem() {
    const blob = new Blob([result], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${prompt.replace(/\s+/g, "_").slice(0, 40)}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function copyToClipboard() {
    navigator.clipboard.writeText(result);
  }

  return (
    <div className="generator">
      <section className="controls">
        <label>
          Model:
          <select
            value={model}
            onChange={(e) => setModel(e.target.value as ModelName)}
          >
            <option value="gpt2">GPT-2 (Transformer)</option>
            <option value="lstm">LSTM</option>
          </select>
        </label>

        <label style={{ width: "100%" }}>
          Title / Prompt:
          {/* Large full-width pill input */}
          <input
            className="prompt-input"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter the poem title or a short prompt"
            aria-label="Poem title or prompt"
          />
          {/* If you want multi-line instead, replace above with:
              <textarea className="prompt-textarea" value={prompt} onChange={...} rows={4} />
          */}
        </label>

        <div className="samples">
          <span style={{ color: "var(--muted)", marginRight: 8 }}>
            Samples:
          </span>
          {samplePrompts.map((s) => (
            <button
              key={s}
              onClick={() => setPrompt(s)}
              type="button"
              className="button-ghost"
            >
              {s}
            </button>
          ))}
        </div>

        <label>
          Max length: {maxLength}
          <input
            type="range"
            min={20}
            max={400}
            value={maxLength}
            onChange={(e) => setMaxLength(Number(e.target.value))}
          />
        </label>

        <label>
          Temperature: {temperature.toFixed(2)}
          <input
            type="range"
            min={0}
            max={1.5}
            step={0.01}
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
          />
        </label>

        <div className="actions">
          <button onClick={onGenerate} disabled={loading} className="primary">
            {loading ? "Generating…" : "Generate"}
          </button>
          <button
            onClick={() => {
              setPrompt("");
              setResult("");
              setError(null);
            }}
            className="button-ghost"
          >
            Clear
          </button>
        </div>
      </section>

      <div
        className="result"
        role="region"
        aria-live="polite"
        aria-label="Generated poem"
      >
        {loading && <div style={{ color: "var(--muted)" }}>Generating…</div>}
        {error && <div className="error">Error: {error}</div>}
        {!loading && !error && result && <pre>{result}</pre>}
        {!loading && !error && !result && (
          <div style={{ color: "var(--muted)" }}>
            No poem yet — press Generate
          </div>
        )}
      </div>

      <div className="meta">
        <div style={{ color: "var(--muted)" }}>
          Model: {model.toUpperCase()}
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          <button
            className="button-ghost"
            onClick={copyToClipboard}
            disabled={!result}
          >
            Copy
          </button>
          <button
            className="button-ghost"
            onClick={downloadPoem}
            disabled={!result}
          >
            Download
          </button>
        </div>
      </div>
    </div>
  );
}
