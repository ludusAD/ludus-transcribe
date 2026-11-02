# Ludus Transcribe

A local-first Streamlit app to transcribe videos offline using Faster-Whisper, with optional AI-powered cleanup via your preferred LLM provider.

**Transcription runs 100% on your machine. No cloud. No API key required to transcribe.**

---

## What it does

1. Upload a video (MP4, MKV, MOV, AVI — up to 2 GB)
2. Split it into chunks (10, 20, or 30 minutes) using FFmpeg
3. Transcribe a chunk offline with Faster-Whisper (runs on CPU, no GPU needed)
4. Optionally polish the raw transcript with an AI model of your choice
5. Download the result as `.txt` or `.md`, browse and search transcript history

---

## Supported LLM providers for polishing

| Provider | What to enter |
|---|---|
| **Local (OpenAI-compatible)** | Base URL of your local server (LM Studio, Ollama, etc.) |
| **OpenAI (ChatGPT)** | Your OpenAI API key |
| **Anthropic (Claude)** | Your Anthropic API key |
| **Google Gemini** | Your Google AI API key |

All fields are blank by default — fill in only the provider you want to use.

---

## Requirements

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and on your PATH

---

## Installation

```bash
git clone https://github.com/ludusAD/ludus-transcribe.git
cd ludus-transcribe

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install your LLM provider SDK if you want transcript polishing:

```bash
pip install anthropic               # Anthropic (Claude)
pip install google-generativeai     # Google Gemini
# OpenAI (ChatGPT) is already included in requirements.txt
```

Verify FFmpeg is available:

```bash
ffmpeg -version
```

---

## Run

```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

---

## Using a local model (no API key needed)

Select **Local (OpenAI-compatible)** in the sidebar. The app connects to your local inference server and fetches available models automatically.

### Option A — Ollama (recommended, simplest)

```bash
# Install (Linux/macOS)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model — Phi-4-mini is a good starting point
# 3.8B parameters · ~2.5 GB download · runs well on 8 GB RAM
ollama pull phi4-mini

# Ollama starts automatically and serves at http://localhost:11434/v1
```

In the app:
- Base URL: `http://localhost:11434/v1`
- API token: leave blank (or type anything — Ollama ignores it)
- Model: select `phi4-mini` from the dropdown

Other good small models: `qwen2.5:7b` (~5 GB, very capable), `mistral` (~4 GB), `llama3.2:3b` (~2 GB, fastest).

### Option B — LM Studio (GUI, beginner-friendly)

1. Download from **https://lmstudio.ai** and install
2. Open LM Studio → **Discover** → search for **Phi-4-Mini**
3. Download the **Q4_K_M** quantized version (~2.5 GB)
4. Go to **Local Server** → click **Start Server** (default port 1234)

In the app:
- Base URL: `http://localhost:1234/v1` (already the default)
- API token: leave blank
- Model: selected automatically from the dropdown

**Recommended model for transcript cleanup:** `Phi-4-mini` — fast, accurate, fits on most laptops.

---

## Project structure

```
app.py                  main Streamlit app
requirements.txt        Python dependencies
.streamlit/
  config.toml           max upload size (2 GB)
uploaded_videos/        uploaded video files (gitignored)
split_segments/         FFmpeg output chunks (gitignored)
transcript_history/     saved transcripts in .txt and .md (gitignored)
```

---

## License

MIT
