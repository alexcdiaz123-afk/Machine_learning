#!/bin/bash

# Render startup script for ML Learning App
# This script ensures proper initialization for production deployment

echo "Starting ML Learning App with Gunicorn..."

# Set environment variables for production
export FLASK_ENV=production
export PYTHONUNBUFFERED=1

# Create necessary directories if they don't exist
mkdir -p static/plots
mkdir -p static/images

# Run Gunicorn with our configuration
gunicorn --config gunicorn_config.py app:app