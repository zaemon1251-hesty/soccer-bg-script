#!/bin/bash

# 1. Download articles
uv run python src/sn_script/download_articles.py

# 2. Extract text
uv run python src/sn_script/extract_text_from_html.py
