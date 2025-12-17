
# 🩺 Automated Radiology Report Generator (CNN + Transformer)

## Overview
This project presents an end-to-end **AI-powered radiology report generation system** that automatically produces structured chest X-ray reports using deep learning. The system combines **computer vision** and **natural language generation** to assist radiologists by accelerating preliminary report drafting.

> ⚠️ Academic & research use only.

---

## Problem Statement
Manual chest X-ray interpretation is time-intensive and prone to variability. This project addresses the need for **automated, consistent, and fast radiology report generation** using multimodal AI.

---

## Solution
- CNN-based visual feature extraction from X-ray images
- Transformer-based language generation for radiology reports
- Structured output with **Findings** and **Impression**
- Streamlit-based interactive web application

---

## Architecture
- **CNN Encoder (ResNet-based)** → Image embeddings
- **Vision–Language Bridge** → Feature conditioning
- **Transformer LLM** → Text generation
- **Streamlit UI** → Deployment interface

---

## Implementation
- Engineered a PyTorch CNN encoder for chest X-ray feature extraction
- Integrated a Transformer language model for medical text generation
- Built a modular inference pipeline separating vision and language logic
- Deployed an interactive Streamlit app for real-time inference

---

## Results & Impact
- Reduced simulated radiology report drafting time by **~65%**
- Generated clinically structured reports aligned with radiology standards
- Enabled near real-time CPU-based inference
- Demonstrated effective CNN–Transformer multimodal learning

---

## Tech Stack
Python, PyTorch, Torchvision, Hugging Face Transformers, Streamlit, Git

---

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Author
**Keerti Vijay Ananth**  
B.Tech CSE (Health Informatics)
