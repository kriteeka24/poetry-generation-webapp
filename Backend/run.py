"""
Simple runner script for the Poetry Generation Backend
"""
from app import app
from config import Config

if __name__ == "__main__":
    print("=" * 60)
    print("  Poetry Generation Backend")
    print("=" * 60)
    print(f"  Host: {Config.HOST}")
    print(f"  Port: {Config.PORT}")
    print(f"  Debug: {Config.DEBUG}")
    print(f"  Fake Mode: {Config.FAKE_MODE}")
    print(f"  Device: {Config.get_device()}")
    print("-" * 60)
    print(f"  GPT-2 Model: {Config.HF_GPT2_MODEL}")
    print(f"  LSTM Model: {Config.HF_LSTM_MODEL}")
    print("=" * 60)
    print(f"\n  Server running at: http://localhost:{Config.PORT}")
    print(f"  API endpoints:")
    print(f"    - GET  /api/health")
    print(f"    - GET  /api/models")
    print(f"    - POST /api/generate")
    print(f"    - POST /api/load-model")
    print(f"    - POST /api/unload-model")
    print("=" * 60 + "\n")
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
