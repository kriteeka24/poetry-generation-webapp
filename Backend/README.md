# Poetry Generation Backend

Flask backend for the Deep Learning Poetry Generation project. Integrates trained LSTM and GPT-2 models from Hugging Face.

## Models

- **GPT-2**: [kriteekathapa/gpt2-poems-finetuned-v1](https://huggingface.co/kriteekathapa/gpt2-poems-finetuned-v1)
  - Fine-tuned GPT-2 transformer model
  - Supports advanced generation parameters (temperature, top-k, top-p, repetition penalty, beam search)

- **LSTM**: [kriteekathapa/lstm-poem-generator-v1](https://huggingface.co/kriteekathapa/lstm-poem-generator-v1)
  - Character-level unidirectional LSTM
  - Supports temperature and top-k sampling

## Setup

### 1. Create Virtual Environment

```bash
cd Backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy the example environment file and adjust settings:

```bash
cp .env.example .env
```

Edit `.env` to configure:

- `FAKE_MODE=0` for real model inference (default)
- `FAKE_MODE=1` for development/testing with fake responses
- `DEVICE=auto` to automatically use GPU if available

### 4. Run the Server

**Development:**

```bash
python run.py
```

**Production (with Gunicorn):**

```bash
gunicorn wsgi:app -w 1 -b 0.0.0.0:5000
```

## API Endpoints

### Health Check

```
GET /api/health
```

### List Models

```
GET /api/models
```

### Generate Poem

```
POST /api/generate
Content-Type: application/json

{
    "model": "gpt2" | "lstm",
    "prompt": "Your poem prompt",
    "max_length": 120,
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 1,
    "do_sample": true
}
```

### Pre-load Model

```
POST /api/load-model
Content-Type: application/json

{
    "model": "gpt2" | "lstm"
}
```

### Unload Model

```
POST /api/unload-model
Content-Type: application/json

{
    "model": "gpt2" | "lstm"
}
```

## Project Structure

```
Backend/
├── app.py              # Main Flask application
├── config.py           # Configuration settings
├── run.py              # Development server runner
├── wsgi.py             # WSGI entry point for production
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment configuration
├── README.md           # This file
└── models/
    ├── __init__.py
    ├── gpt2_generator.py   # GPT-2 model wrapper
    └── lstm_generator.py   # LSTM model wrapper
```

## Connecting to Frontend

Update the Frontend's API configuration to point to this backend:

1. In development, update `Frontend/vite.config.ts` proxy settings
2. Or set the API base URL in `Frontend/src/api.ts`

Example vite.config.ts proxy:

```typescript
export default defineConfig({
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:5000",
        changeOrigin: true,
      },
    },
  },
});
```

## Hardware Requirements

- **CPU**: Works on CPU, but slower for GPT-2
- **GPU (Recommended)**: NVIDIA GPU with CUDA support for faster inference
- **RAM**: At least 8GB (16GB recommended for loading both models)
- **Disk**: ~1GB for model downloads

## First Run

On first run, models will be automatically downloaded from Hugging Face Hub.
This may take a few minutes depending on your internet connection.

## Troubleshooting

### CUDA Out of Memory

- Try using CPU: Set `DEVICE=cpu` in `.env`
- Or reduce `max_length` parameter

### Model Loading Fails

- Check internet connection for model download
- Verify Hugging Face model URLs are accessible
- Check if you have enough disk space

### Slow Generation

- Use GPU if available
- Reduce `max_length` for faster responses
- Pre-load models using `/api/load-model` endpoint
