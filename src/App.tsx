import React from "react";
import Generator from "./components_Generator";

export default function App() {
  return (
    <div className="app">
      <header className="header">
        <h1>Poetry generation using deep learning models</h1>
        <p className="subtitle">
          LSTM vs GPT-2 — generate and compare poetic text
        </p>
      </header>

      <main>
        <Generator />
      </main>

      <footer className="footer">
        <small>Deep learning approaches for poetry generation</small>
      </footer>
    </div>
  );
}
