# ObsidianNotesAgent

A local AI agent that crawls your Obsidian vault, builds a vector database of your notes, and lets you ask questions, find connections, categorize content, identify knowledge gaps, and get actionable suggestions — all powered by Ollama and running entirely on your machine.

## What It Does

- Scans your Obsidian vault (or a subfolder) for markdown notes
- Extracts clean content, metadata (tags, backlinks, etc.), and frontmatter
- Chunks notes and embeds them using Ollama embedding models (e.g. nomic-embed-text, bge-m3)
- Stores everything in a persistent Chroma vector database
- Provides a RAG (Retrieval-Augmented Generation) interface to query your notes with Ollama LLMs
- Supports:
  - Semantic search & retrieval
  - Insight generation (connections, gaps, suggestions)
  - Re-indexing changed notes
  - Customizable prompts for different reasoning styles

Goal: Turn your personal Obsidian knowledge base into a living, queryable second brain.

## Features

- Local-only (Ollama + Chroma + your GPU/CPU)
- Incremental vault updates (add/re-index changed notes)
- Flexible prompt templates (swap for summarization, gap-finding, etc.)
- Retrieval with configurable k and MMR (diversity mode)
- Easy to extend into a continuous agent or TUI

## Requirements

- Python 3.11+
- Ollama installed and running (`ollama serve`)
  - Recommended models:
    - LLM: `qwen3:14b` or `llama3.2:8b` (or any chat model)
    - Embedding: `nomic-embed-text` (default) or `bge-m3`
- GPU (AMD RX 6950 XT or similar) with ROCm for acceleration (optional but highly recommended)
- Enough VRAM (≥10–12 GB for 14B models at good quantization)

## Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/drithird/ObsidianNotesAgent.git
   cd ObsidianNotesAgent
2. Configuration/Main Loop can be found in src/obsidiannoteagent/core/main.py

This tool would not be possible without the obsidiantools project which can be found [here](https://github.com/mfarragher/obsidiantools)

Example vault provided by the CEO of Obsidian Steph Ango learn more about how to use it [here](https://stephango.com/vault)
