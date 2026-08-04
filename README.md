# VISION MASK

This repository is organized for an image/vision machine learning workflow with a clear separation between data, model artifacts, source code, and experimentation.

## Project Structure

- `dataset/` — input datasets and processed data files
- `models/` — trained model checkpoints and saved artifacts
- `src/` — core implementation code
- `notebooks/` — exploratory notebooks and analysis experiments
- `requirements.txt` — Python dependencies

## Suggested Workflow

1. Place datasets in `dataset/`
2. Implement training/inference code in `src/`
3. Save trained models in `models/`
4. Use notebooks for experimentation and visualization

## Setup

```bash
pip install -r requirements.txt
```
