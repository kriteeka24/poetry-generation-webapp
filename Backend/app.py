"""
Flask Backend for Poetry Generation
Integrates GPT-2 and LSTM models from Hugging Face for poem generation.

Models:
- GPT-2: kriteekathapa/gpt2-poems-finetuned-v1
- LSTM: kriteekathapa/lstm-poem-generator-v1
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
import logging
from config import Config
from models import GPT2Generator, LSTMGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Model instances (lazy loaded)
_gpt2_generator = None
_lstm_generator = None


def get_gpt2_generator() -> GPT2Generator:
    """Get or create GPT-2 generator instance"""
    global _gpt2_generator
    if _gpt2_generator is None:
        _gpt2_generator = GPT2Generator(
            model_name=Config.HF_GPT2_MODEL,
            device=Config.get_device(),
            cache_dir=Config.MODEL_CACHE_DIR,
            hf_token=Config.HF_TOKEN
        )
    return _gpt2_generator


def get_lstm_generator() -> LSTMGenerator:
    """Get or create LSTM generator instance"""
    global _lstm_generator
    if _lstm_generator is None:
        _lstm_generator = LSTMGenerator(
            model_name=Config.HF_LSTM_MODEL,
            device=Config.get_device(),
            cache_dir=Config.MODEL_CACHE_DIR,
            hf_token=Config.HF_TOKEN
        )
    return _lstm_generator


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "fake_mode": Config.FAKE_MODE,
        "device": Config.get_device(),
        "models": {
            "gpt2": Config.HF_GPT2_MODEL,
            "lstm": Config.HF_LSTM_MODEL
        }
    })


@app.route("/api/models", methods=["GET"])
def list_models():
    """List available models and their status"""
    gpt2 = get_gpt2_generator()
    lstm = get_lstm_generator()
    
    return jsonify({
        "models": [
            {
                "id": "gpt2",
                "name": "GPT-2 (Transformer)",
                "description": "Fine-tuned GPT-2 model for poetry generation",
                "hf_model": Config.HF_GPT2_MODEL,
                "loaded": gpt2.is_loaded(),
                "supported_params": ["max_length", "temperature", "top_k", "top_p", "repetition_penalty",num_return_sequences]
            },
            {
                "id": "lstm",
                "name": "LSTM (Recurrent Neural Network)",
                "description": "Character-level LSTM model for poetry generation",
                "hf_model": Config.HF_LSTM_MODEL,
                "loaded": lstm.is_loaded(),
                "supported_params": ["max_length", "temperature", "top_k"]
            }
        ]
    })


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Generate poetry based on the given prompt.
    
    Request body:
    {
        "model": "gpt2" | "lstm",
        "prompt": "Your poem prompt",
        "max_length": 200,
        "temperature": 1,
        "top_k": 50,
        "top_p": 0.92,
        "repetition_penalty": 1.1,
        "num_beams": 1,
        "do_sample": true,
        "num_return_sequences": 1
    }
    
    Response:
    {
        "generated_text": "Generated poem...",
        "generation_time": 1.23,
        "model": "gpt2",
        "parameters": {...}
    }
    """
    data = request.get_json() or {}
    
    # Extract parameters
    model = data.get("model")
    prompt = data.get("prompt", "").strip()
    max_length = int(data.get("max_length", 256))
    temperature = float(data.get("temperature", 0.80))
    top_k = int(data.get("top_k", 50))
    top_p = float(data.get("top_p", 0.92))
    repetition_penalty = float(data.get("repetition_penalty", 1.1))
    num_return_sequences = int(data.get("num_return_sequences", 1))
    

    
    # Validate required parameters
    if not model:
        return jsonify({"error": "Missing 'model' in request body"}), 400
    if not prompt:
        return jsonify({"error": "Missing 'prompt' in request body"}), 400
    if model not in ["gpt2", "lstm"]:
        return jsonify({"error": f"Unsupported model: {model}. Use 'gpt2' or 'lstm'"}), 400
    
    # Development mode: return fake response
    if Config.FAKE_MODE:
        logger.info(f"FAKE_MODE: Generating fake poem for model={model}")
        fake_poem = generate_fake_poem(prompt, model, max_length, temperature, top_k, top_p)
        time.sleep(0.5)  # Simulate processing time
        return jsonify({
            "generated_text": fake_poem,
            "generation_time": 0.5,
            "model": model,
            "fake_mode": True,
            "parameters": {
                "max_length": max_length,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p
            }
        })
    
    # Real generation
    start_time = time.time()
    
    try:
        if model == "gpt2":
            generator = get_gpt2_generator()
            generated_text = generator.generate(
                prompt=f"Title: {prompt}\n\n",
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences
            )
        else:  # lstm
            generator = get_lstm_generator()
            generated_text = generator.generate(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        generation_time = time.time() - start_time
        logger.info(f"Generated poem with {model} in {generation_time:.2f}s")
        
        return jsonify({
            "generated_text": generated_text,
            "generation_time": round(generation_time, 3),
            "model": model,
            "parameters": {
                "max_length": max_length,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p if model == "gpt2" else None,
                "repetition_penalty": repetition_penalty if model == "gpt2" else None,
                "num_return_sequences": num_return_sequences if model == "gpt2" else None
            }
        })
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return jsonify({
            "error": f"Generation failed: {str(e)}"
        }), 500


@app.route("/api/load-model", methods=["POST"])
def load_model():
    """
    Pre-load a model into memory.
    
    Request body:
    {
        "model": "gpt2" | "lstm"
    }
    """
    data = request.get_json() or {}
    model = data.get("model")
    
    if not model or model not in ["gpt2", "lstm"]:
        return jsonify({"error": "Invalid model. Use 'gpt2' or 'lstm'"}), 400
    
    try:
        start_time = time.time()
        
        if model == "gpt2":
            generator = get_gpt2_generator()
            generator.load()
        else:
            generator = get_lstm_generator()
            generator.load()
        
        load_time = time.time() - start_time
        
        return jsonify({
            "success": True,
            "model": model,
            "load_time": round(load_time, 3),
            "message": f"{model.upper()} model loaded successfully"
        })
        
    except Exception as e:
        logger.error(f"Failed to load model {model}: {e}")
        return jsonify({
            "error": f"Failed to load model: {str(e)}"
        }), 500


@app.route("/api/unload-model", methods=["POST"])
def unload_model():
    """
    Unload a model from memory to free resources.
    
    Request body:
    {
        "model": "gpt2" | "lstm"
    }
    """
    data = request.get_json() or {}
    model = data.get("model")
    
    if not model or model not in ["gpt2", "lstm"]:
        return jsonify({"error": "Invalid model. Use 'gpt2' or 'lstm'"}), 400
    
    try:
        if model == "gpt2":
            generator = get_gpt2_generator()
            generator.unload()
        else:
            generator = get_lstm_generator()
            generator.unload()
        
        return jsonify({
            "success": True,
            "model": model,
            "message": f"{model.upper()} model unloaded successfully"
        })
        
    except Exception as e:
        logger.error(f"Failed to unload model {model}: {e}")
        return jsonify({
            "error": f"Failed to unload model: {str(e)}"
        }), 500


def generate_fake_poem(prompt: str, model: str, max_length: int, temperature: float, top_k: int, top_p: float) -> str:
    """Generate a fake poem for development/testing"""
    return f"""{prompt}

(Generated by {model.upper()} — FAKE MODE)
Parameters: len={max_length}, temp={temperature:.2f}, top_k={top_k}, top_p={top_p:.2f}

Softly the sunrise slides over fields of code,
Lines like rivers, memory in ode.
A gentle rhyme of bytes and dreams,
Whispers of logic in silver beams.

Through digital realms where thoughts take flight,
The neural pathways dance in light.
Each token born from probability's embrace,
Creating verses with algorithmic grace.

In the quiet hum of silicon minds,
Poetry emerges, one of many kinds.
A fusion of art and machine's design,
Where human creativity and AI intertwine."""


# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    logger.info(f"Starting Poetry Generation Backend")
    logger.info(f"  - Host: {Config.HOST}")
    logger.info(f"  - Port: {Config.PORT}")
    logger.info(f"  - Debug: {Config.DEBUG}")
    logger.info(f"  - Fake Mode: {Config.FAKE_MODE}")
    logger.info(f"  - Device: {Config.get_device()}")
    logger.info(f"  - GPT-2 Model: {Config.HF_GPT2_MODEL}")
    logger.info(f"  - LSTM Model: {Config.HF_LSTM_MODEL}")
    
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    )
