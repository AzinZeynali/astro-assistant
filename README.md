---
title: Astronomy Assistant
emoji: 🔭
colorFrom: indigo
colorTo: purple
sdk: static
app_file: index.html
pinned: false
---

# 🔭 Astronomy Assistant

**Live demo:** https://huggingface.co/spaces/AzinZeynali/astro-assistant

A browser-based app for exploring exoplanets. Everything runs client-side in JavaScript, no server needed.

## Features

1. **Ask a question:** semantic search over Wikipedia astronomy articles using sentence embeddings (all-MiniLM-L6-v2) and cosine similarity.
2. **Is it a planet?** a Random Forest classifier trained on NASA's Kepler Objective of Interest table predicts whether a signal is a confirmed planet or a false alarm.

## How it works

- `fetch_data.py` downloads the articles and the NASA Kepler table.
- `build_knowledge.py` precomputes the article embeddings (`knowledge.json`).
- `train_classifier.py` trains the Random Forest; the exported model runs in the browser via `forest.js`.
- `index.html` and `app.js` provide the interface.

## Run locally

```bash
pip install -r requirements.txt
python fetch_data.py
python build_knowledge.py
python train_classifier.py
```

Then open `index.html` in a browser.

## Data sources

- Wikipedia (CC BY-SA 4.0)
- NASA Exoplanet Archive (Kepler cumulative table)

Built by Azin Zeynali.