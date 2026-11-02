import os
import subprocess
import tempfile
from datetime import date, datetime

import streamlit as st
from faster_whisper import WhisperModel
from openai import OpenAI

HOSTED = os.environ.get("HOSTED", "").lower() in ("1", "true", "yes")

st.set_page_config(page_title="Ludus Transcribe", layout="wide")
st.title("Ludus Transcribe")
st.caption("Offline video transcription with optional AI polishing")

# ---------------------------------------------------------------------------
# Directories — per-session temp dirs in hosted mode, persistent dirs locally
# ---------------------------------------------------------------------------
if HOSTED:
    if "work_dir" not in st.session_state:
        st.session_state["work_dir"] = tempfile.mkdtemp(prefix="ludus_transcribe_")
    _base = st.session_state["work_dir"]
    UPLOAD_DIR = os.path.join(_base, "uploaded_videos")
    SPLIT_DIR = os.path.join(_base, "split_segments")
    HISTORY_DIR = os.path.join(_base, "transcript_history")
else:
    UPLOAD_DIR = "uploaded_videos"
    SPLIT_DIR = "split_segments"
    HISTORY_DIR = "transcript_history"

for d in (UPLOAD_DIR, SPLIT_DIR, HISTORY_DIR):
    os.makedirs(d, exist_ok=True)

if HOSTED:
    st.info(
        "🔒 **Hosted mode** — your files are processed in memory on this server and deleted when your session ends. "
        "Nothing is stored, logged, or visible to anyone — including the server operator.",
        icon=None,
    )

# ---------------------------------------------------------------------------
# LLM provider config (sidebar)
# ---------------------------------------------------------------------------
PROVIDERS = ["Local (OpenAI-compatible)", "OpenAI (ChatGPT)", "Anthropic (Claude)", "Google Gemini"]

PROVIDER_DEFAULTS = {
    "Local (OpenAI-compatible)": {
        "base_url": "http://localhost:1234/v1",
        "models": [],
    },
    "OpenAI (ChatGPT)": {
        "base_url": "https://api.openai.com/v1",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    },
    "Anthropic (Claude)": {
        "base_url": "",
        "models": [
            "claude-opus-4-5",
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
        ],
    },
    "Google Gemini": {
        "base_url": "",
        "models": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
    },
}

st.sidebar.header("AI Polishing — Provider")
provider = st.sidebar.selectbox("Provider", PROVIDERS)

api_key = ""
base_url = ""
model_options = PROVIDER_DEFAULTS[provider]["models"]

if provider == "Local (OpenAI-compatible)":
    base_url = st.sidebar.text_input(
        "Base URL", value=PROVIDER_DEFAULTS["Local (OpenAI-compatible)"]["base_url"]
    )
    api_key = st.sidebar.text_input("API token (optional)", type="password")
    if base_url:
        _key = api_key.strip() or "no-key"
        try:
            _client = OpenAI(base_url=base_url, api_key=_key)
            model_options = [m.id for m in _client.models.list().data]
        except Exception:
            model_options = []

elif provider == "OpenAI (ChatGPT)":
    api_key = st.sidebar.text_input("OpenAI API key", type="password")

elif provider == "Anthropic (Claude)":
    api_key = st.sidebar.text_input("Anthropic API key", type="password")

elif provider == "Google Gemini":
    api_key = st.sidebar.text_input("Gemini API key", type="password")

if model_options:
    selected_model = st.sidebar.selectbox("Model", model_options)
else:
    selected_model = st.sidebar.text_input("Model name", value="local-model")


# ---------------------------------------------------------------------------
# LLM call abstraction
# ---------------------------------------------------------------------------
def call_llm(system_msg: str, user_msg: str) -> str:
    if provider in ("Local (OpenAI-compatible)", "OpenAI (ChatGPT)"):
        _base = base_url if provider == "Local (OpenAI-compatible)" else None
        _key = api_key.strip() or "no-key"
        kwargs = dict(api_key=_key)
        if _base:
            kwargs["base_url"] = _base
        client = OpenAI(**kwargs)
        resp = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content

    elif provider == "Anthropic (Claude)":
        try:
            import anthropic
        except ImportError:
            raise RuntimeError("Run: pip install anthropic")
        client = anthropic.Anthropic(api_key=api_key.strip())
        msg = client.messages.create(
            model=selected_model,
            max_tokens=4096,
            system=system_msg,
            messages=[{"role": "user", "content": user_msg}],
        )
        return msg.content[0].text

    elif provider == "Google Gemini":
        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError("Run: pip install google-generativeai")
        genai.configure(api_key=api_key.strip())
        gemini = genai.GenerativeModel(selected_model, system_instruction=system_msg)
        return gemini.generate_content(user_msg).text

    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_transcript(segment_name: str, text: str) -> tuple[str, str]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(segment_name)[0]
    name = f"{ts}_{base}_transcript"
    txt = os.path.join(HISTORY_DIR, f"{name}.txt")
    md = os.path.join(HISTORY_DIR, f"{name}.md")
    for path in (txt, md):
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return txt, md


def parse_dt(filename: str) -> datetime | None:
    parts = filename.split("_", 2)
    if len(parts) < 2:
        return None
    try:
        return datetime.strptime(f"{parts[0]}_{parts[1]}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Tabs — History hidden in hosted mode (no persistent storage)
# ---------------------------------------------------------------------------
if HOSTED:
    transcribe_tab = st.container()
    history_tab = None
else:
    transcribe_tab, history_tab = st.tabs(["Transcribe", "History"])

with transcribe_tab:
    uploaded_file = st.sidebar.file_uploader(
        "Upload a video", type=["mp4", "mkv", "mov", "avi"]
    )

    if uploaded_file:
        video_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.sidebar.success(f"Loaded: {uploaded_file.name}")

        # Step 1 — split
        st.header("Step 1 — Split video into chunks")
        split_minutes = st.selectbox("Chunk duration (minutes)", [10, 20, 30])

        if st.button("Split video"):
            with st.spinner("Splitting with FFmpeg…"):
                for f in os.listdir(SPLIT_DIR):
                    os.remove(os.path.join(SPLIT_DIR, f))
                subprocess.run(
                    [
                        "ffmpeg", "-i", video_path,
                        "-c", "copy", "-map", "0",
                        "-segment_time", str(split_minutes * 60),
                        "-f", "segment", "-reset_timestamps", "1",
                        os.path.join(SPLIT_DIR, "part_%03d.mp4"),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            st.success(f"Split into {split_minutes}-minute chunks.")

        # Step 2 — pick segment
        segments = sorted(f for f in os.listdir(SPLIT_DIR) if f.endswith(".mp4"))
        if segments:
            st.header("Step 2 — Pick a segment")
            selected_segment = st.selectbox("Chunk to transcribe", segments)
            segment_path = os.path.join(SPLIT_DIR, selected_segment)
            audio_path = os.path.join(SPLIT_DIR, "temp_audio.wav")

            if st.button("Extract audio & transcribe"):
                with st.spinner("Extracting audio…"):
                    if os.path.exists(audio_path):
                        os.remove(audio_path)
                    subprocess.run(
                        [
                            "ffmpeg", "-i", segment_path,
                            "-vn", "-acodec", "pcm_s16le",
                            "-ar", "16000", "-ac", "1",
                            audio_path,
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

                st.header("Step 3 — Transcribing offline")
                with st.spinner("Running Faster-Whisper on CPU (100% offline)…"):
                    model = WhisperModel("base", device="cpu", compute_type="int8")
                    segments_gen, _ = model.transcribe(audio_path, beam_size=5)
                    lines = []
                    for seg in segments_gen:
                        ts = f"[{int(seg.start // 60):02d}:{int(seg.start % 60):02d}]"
                        line = f"{ts} {seg.text}"
                        lines.append(line)
                        st.text(line)
                    raw = "\n".join(lines)

                st.session_state["raw_transcript"] = raw
                st.session_state["active_segment"] = selected_segment

                if HOSTED:
                    st.success("Transcription complete. Choose what to do in Step 4.")
                else:
                    save_transcript(f"raw_{selected_segment}", raw)
                    st.success("Transcription complete. Raw transcript saved to history.")

            # Step 4 — AI polish or raw download
            if "raw_transcript" in st.session_state:
                st.header("Step 4 — Output")
                polish_mode = st.radio(
                    "What would you like to do?",
                    ["Download raw transcript", "Polish with AI"],
                    horizontal=True,
                )

                if polish_mode == "Download raw transcript":
                    raw = st.session_state["raw_transcript"]
                    st.subheader("Raw transcript")
                    st.text_area("Preview", value=raw, height=300)
                    st.download_button(
                        "Download .txt", raw,
                        file_name="transcript.txt", mime="text/plain",
                    )
                    st.download_button(
                        "Download .md", raw,
                        file_name="transcript.md", mime="text/markdown",
                    )

                else:
                    instruction = st.text_area(
                        "Instructions for the AI",
                        value=(
                            "Clean up the grammar of this transcript. Remove filler words "
                            "(um, uh, like) and organise the text into readable paragraphs "
                            "with logical headings. Do not change the meaning."
                        ),
                    )

                    if st.button("Polish transcript"):
                        with st.spinner(f"Sending to {provider} — {selected_model}…"):
                            try:
                                polished = call_llm(
                                    system_msg="You are an expert editor who cleans up raw audio transcripts.",
                                    user_msg=f"{instruction}\n\nTranscript:\n{st.session_state['raw_transcript']}",
                                )
                                if not HOSTED:
                                    save_transcript(
                                        st.session_state.get("active_segment", "segment"),
                                        polished,
                                    )
                                st.subheader("Polished transcript")
                                st.write(polished)
                                st.download_button(
                                    "Download .txt", polished,
                                    file_name="transcript.txt", mime="text/plain",
                                )
                                st.download_button(
                                    "Download .md", polished,
                                    file_name="transcript.md", mime="text/markdown",
                                )
                            except Exception as e:
                                st.error(f"AI polishing failed: {e}")

if history_tab is not None:
    with history_tab:
        st.header("Transcript history")
        files = sorted(
            [f for f in os.listdir(HISTORY_DIR) if f.endswith(".md")], reverse=True
        )

        if not files:
            st.info("No saved transcripts yet.")
        else:
            records = [{"file": f, "dt": parse_dt(f)} for f in files]
            query = st.text_input("Search by filename").strip().lower()

            valid_dates = sorted({r["dt"].date() for r in records if r["dt"]})
            if valid_dates:
                date_range = st.date_input(
                    "Date range",
                    value=(min(valid_dates), max(valid_dates)),
                    min_value=min(valid_dates),
                    max_value=max(valid_dates),
                )
                start_date = date_range[0] if isinstance(date_range, tuple) else date_range
                end_date = date_range[1] if isinstance(date_range, tuple) and len(date_range) == 2 else start_date
            else:
                start_date, end_date = date.min, date.max

            filtered = [
                r["file"] for r in records
                if (not query or query in r["file"].lower())
                and (r["dt"] is None or start_date <= r["dt"].date() <= end_date)
            ]

            st.caption(f"Showing {len(filtered)} of {len(files)} transcripts")
            if not filtered:
                st.warning("No transcripts match the filters.")
                st.stop()

            chosen = st.selectbox("Transcript", filtered)
            with open(os.path.join(HISTORY_DIR, chosen), encoding="utf-8") as f:
                content = f.read()

            st.text_area("Content", value=content, height=400)
            st.download_button(
                "Download .md", content, file_name=chosen, mime="text/markdown"
            )
