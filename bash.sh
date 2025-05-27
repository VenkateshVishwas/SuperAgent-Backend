#!/bin/bash

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Installing Node.js dependencies..."
npm install

echo "Downloading model zip from Google Drive..."
curl -L -o distilbert_super_agent.zip "https://drive.google.com/uc?export=download&id=1RGGn104MHeWF12Uxw48g8Y-0Tx4_cJdL"

echo "Unzipping model files..."
unzip -o distilbert_super_agent.zip

echo "✅ Model ready"


