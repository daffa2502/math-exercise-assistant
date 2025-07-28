# Qwen3-1.7B Model Setup

This guide explains how to download and set up the Qwen3-1.7B model locally for the Math Exercise Assistant.

## Prerequisites

- Python 3.8+
- At least 4GB of free disk space
- Internet connection for initial download

## Setup Steps

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the Model**
   ```bash
   python scripts/download_model.py
   ```

3. **Verify Installation**
   The model should be downloaded to `models/qwen3-1.7b/` directory.

4. **Update Environment**
   Set `MOCK_MODE=false` in your `.env` file to use the actual model.

## Model Information

- **Model**: Qwen3-1.7B
- **Size**: ~1.7 billion parameters
- **Storage**: ~3.4GB on disk
- **Type**: Causal Language Model
- **Use Case**: Math problem generation and explanation

## Troubleshooting

- If download fails, check your internet connection and try again
- If you encounter CUDA errors, the model will automatically fall back to CPU
- For memory issues, ensure you have at least 4GB RAM available
