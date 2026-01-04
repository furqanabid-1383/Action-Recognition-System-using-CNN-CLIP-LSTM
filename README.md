**-> Action Recognition API — CNN + LSTM Architecture**

This project implements an Action Recognition and Image Captioning system using a deep learning CNN + LSTM architecture, developed to meet academic coursework requirements.
The system takes an input image, predicts the human action occurring in the image, and generates a natural-language caption describing the action.

**-> System Architecture**

The project follows a **two-stage deep learning pipeline**:

**1️.CNN – Feature Extraction & Action Classification**

**Model**: CLIP (ViT-B/32)
Extracts high-level visual features from images
Uses cosine similarity between image embeddings and action text prompts
Predicts the most likely human action with confidence scores

**2️. LSTM – Caption Generation**

**Model**: 2-Layer LSTM (PyTorch)
Takes CNN feature vectors as input
Generates captions word-by-word
Demonstrates sequence modeling using recurrent neural networks

**-> Pipeline**:
Image → CNN (CLIP) → Feature Vector → Action Classification → LSTM Caption Generation

**-> Tech Stack**

Following is the technology stack for this project:

**1. Backend**
FastAPI – REST API
PyTorch – Deep learning framework
CLIP (ViT-B/32) – CNN for visual feature extraction
LSTM – Sequential caption generation
Uvicorn – ASGI server

**2. Frontend**
React (Vite) – User interface
Fetch / Axios – API communication
Modern UI for image upload and result display
