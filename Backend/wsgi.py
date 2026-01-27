"""
WSGI entry point for production deployment (e.g., with Gunicorn)

Usage:
    gunicorn wsgi:app -w 1 -b 0.0.0.0:5000

Note: Use 1 worker (-w 1) to share model instances across requests.
For multiple workers, each worker will load its own model instance.
"""
from app import app

if __name__ == "__main__":
    app.run()
