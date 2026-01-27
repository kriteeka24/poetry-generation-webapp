import React from "react";
import Generator from "./components_Generator";

export default function App() {
  return (
    <div className="app">
      <header className="header">
        <h1>Deep Learning Approaches for Poetry Generation</h1>
        <p className="subtitle">
          Comparative Analysis: LSTM vs Transformer (GPT-2)
        </p>
      </header>

      <main>
        <Generator />
      </main>
    </div>
  );
}
