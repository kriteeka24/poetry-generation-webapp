import React, { useState } from "react";
import { generatePoem } from "./api";

type ModelName = "gpt2" | "lstm";

export default function Generator() {
  const [model, setModel] = useState<ModelName>("gpt2");
  const [prompt, setPrompt] = useState("A Quiet Morning");
  const [maxLength, setMaxLength] = useState(256);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(50);
  const [topP, setTopP] = useState(0.92);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.1);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [generationTime, setGenerationTime] = useState<number | null>(null);

  const samplePrompts = [
    "A Quiet Morning",
    "Lonely River",
    "The Last Sunset",
    "Whispers of the Old Oak",
    "Moonlit Dreams",
    "Autumn Leaves",
  ];

  async function onGenerate() {
    setError(null);
    setResult("");
    setLoading(true);
    setGenerationTime(null);
    const startTime = Date.now();
    try {
      // Build request params based on model type
      const params: any = {
        model,
        prompt,
        max_length: maxLength,
        temperature,
        top_k: topK,
        top_p: topP,
      };

      // Add GPT-2 specific parameters
      if (model === "gpt2") {
        params.repetition_penalty = repetitionPenalty;
      }

      const res = await generatePoem(params);
      setResult(res.generated_text ?? "");
      setGenerationTime((Date.now() - startTime) / 1000);
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

  function resetToDefaults() {
    setTemperature(0.8);
    setTopK(50);
    setTopP(0.92);
    setRepetitionPenalty(1.1);
    setMaxLength(256);
  }

  return (
    <div className="generator">
      {/* Model Selection */}
      <section className="model-selection">
        <h3 className="section-title">Select Model</h3>
        <select
          className="model-select"
          value={model}
          onChange={(e) => setModel(e.target.value as "gpt2" | "lstm")}
        >
          <option value="gpt2">GPT-2 (Transformer) — Recommended</option>
          <option value="lstm">LSTM (Recurrent Neural Network)</option>
        </select>
      </section>

      {/* Prompt Section */}
      <section className="prompt-section">
        <h3 className="section-title">Input Prompt</h3>
        <textarea
          className="prompt-textarea"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter a poem title, theme, or starting lines..."
          rows={2}
        />
        <div className="samples">
          <span className="samples-label">Quick prompts:</span>
          <div className="samples-buttons">
            {samplePrompts.map((s) => (
              <button
                key={s}
                onClick={() => setPrompt(s)}
                type="button"
                className={`sample-btn ${prompt === s ? "active" : ""}`}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* Generation Parameters */}
      <section className="params-section">
        <h3 className="section-title">Generation Parameters</h3>

        <div className="params-grid">
          <div className="param-card">
            <div className="param-header">
              <label>Max Length</label>
              <span className="param-value">{maxLength}</span>
            </div>
            <input
              type="range"
              min={20}
              max={500}
              value={maxLength}
              onChange={(e) => setMaxLength(Number(e.target.value))}
              className="slider"
            />
            <div className="param-hint">Number of tokens to generate</div>
          </div>

          <div className="param-card">
            <div className="param-header">
              <label>Temperature</label>
              <span className="param-value">{temperature.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.1}
              max={2.0}
              step={0.01}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              className="slider"
            />
            <div className="param-hint">
              {temperature < 0.5
                ? "Low: More focused and deterministic"
                : temperature < 1.0
                  ? "Medium: Balanced creativity"
                  : "High: More creative and diverse"}
            </div>
          </div>

          <div className="param-card">
            <div className="param-header">
              <label>Top-P (Nucleus)</label>
              <span className="param-value">{topP.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min={0.1}
              max={1.0}
              step={0.01}
              value={topP}
              onChange={(e) => setTopP(Number(e.target.value))}
              className="slider"
            />
            <div className="param-hint">Cumulative probability threshold</div>
          </div>

          <div className="param-card">
            <div className="param-header">
              <label>Top-K</label>
              <span className="param-value">{topK}</span>
            </div>
            <input
              type="range"
              min={1}
              max={100}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="slider"
            />
            <div className="param-hint">Number of top tokens to consider</div>
          </div>
        </div>

        {/* Advanced Parameters Toggle - Only for GPT-2 */}
        {model === "gpt2" && (
          <button
            className="advanced-toggle"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            {showAdvanced ? "▲ Hide" : "▼ Show"} Advanced Options
          </button>
        )}

        {model === "gpt2" && showAdvanced && (
          <div className="advanced-params">
            <div className="params-grid">
              <div className="param-card">
                <div className="param-header">
                  <label>Repetition Penalty</label>
                  <span className="param-value">
                    {repetitionPenalty.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min={1.0}
                  max={2.0}
                  step={0.05}
                  value={repetitionPenalty}
                  onChange={(e) => setRepetitionPenalty(Number(e.target.value))}
                  className="slider"
                />
                <div className="param-hint">
                  Penalize repeated tokens (1.0 = no penalty)
                </div>
              </div>
            </div>

            <button className="reset-btn" onClick={resetToDefaults}>
              ↺ Reset to Defaults
            </button>
          </div>
        )}
      </section>

      {/* Actions */}
      <div className="actions">
        <button
          onClick={onGenerate}
          disabled={loading || !prompt.trim()}
          className="generate-btn"
        >
          {loading ? (
            <>
              <span className="spinner"></span> Generating...
            </>
          ) : (
            "Generate Poetry"
          )}
        </button>
        <button
          onClick={() => {
            setPrompt("");
            setResult("");
            setError(null);
            setGenerationTime(null);
          }}
          className="clear-btn"
        >
          Clear All
        </button>
      </div>

      {/* Result Section */}
      <div className="result-section">
        <div className="result-header">
          <h3>Output</h3>
          {generationTime !== null && (
            <span className="generation-time">
              Generated in {generationTime.toFixed(2)}s
            </span>
          )}
        </div>
        <div
          className="result"
          role="region"
          aria-live="polite"
          aria-label="Generated poem"
        >
          {loading && (
            <div className="loading-state">
              <div className="loading-animation">
                <div className="dot"></div>
                <div className="dot"></div>
                <div className="dot"></div>
              </div>
              <p>Crafting your poem...</p>
            </div>
          )}
          {error && (
            <div className="error-state">
              <p>Error: {error}</p>
            </div>
          )}
          {!loading && !error && result && (
            <pre className="poem-text">{result}</pre>
          )}
          {!loading && !error && !result && (
            <div className="empty-state">
              <p>Generated poetry will appear here</p>
              <small>Select a model, enter a prompt, and click Generate</small>
            </div>
          )}
        </div>

        {result && (
          <div className="result-actions">
            <button className="action-btn" onClick={copyToClipboard}>
              Copy to Clipboard
            </button>
            <button className="action-btn" onClick={downloadPoem}>
              Download as Text
            </button>
          </div>
        )}
      </div>

      {/* Current Settings Summary */}
      <div className="settings-summary">
        <span className="summary-item">
          <strong>Model:</strong> {model.toUpperCase()}
        </span>
        <span className="summary-item">
          <strong>Temp:</strong> {temperature.toFixed(2)}
        </span>
        <span className="summary-item">
          <strong>Top-P:</strong> {topP.toFixed(2)}
        </span>
        <span className="summary-item">
          <strong>Top-K:</strong> {topK}
        </span>
        <span className="summary-item">
          <strong>Length:</strong> {maxLength}
        </span>
      </div>
    </div>
  );
}
