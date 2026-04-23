# HOW TO RUN:
# 1. pip install -r requirements.txt
# 2. streamlit run app.py

import logging
import streamlit as st

from inference import summarize_text, translate_to_hindi, translate_to_spanish
from project_paths import load_config, resolve_asset

st.set_page_config(
    page_title="LinguaVerse AI · Yug Nagda",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, .stApp {
    background: radial-gradient(ellipse at 15% 10%, #0d1b4b 0%, #060612 60%, #0a0a18 100%);
    color: #d4d8f0;
    font-family: 'Inter', sans-serif;
}

section[data-testid="stSidebar"] {
    background: rgba(8,10,30,0.95) !important;
    backdrop-filter: blur(20px);
    border-right: 1px solid rgba(99,102,241,0.2);
}
section[data-testid="stSidebar"] * { color: #d4d8f0 !important; }

.hero-banner {
    background: linear-gradient(135deg, rgba(99,102,241,0.18), rgba(16,185,129,0.10), rgba(245,101,175,0.12));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 2.4rem 2.8rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #6366f1, #10b981, #f565af, #6366f1);
    background-size: 300%;
    animation: shimmer 4s linear infinite;
}
@keyframes shimmer { to { background-position: 300% 0; } }

.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, #a5b4fc, #34d399, #f9a8d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.4rem;
}
.hero-sub { font-size: 0.88rem; color: rgba(212,216,240,0.55); letter-spacing: 0.04em; }
.badge {
    display: inline-block;
    padding: 0.2rem 0.75rem;
    border-radius: 20px;
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.05em;
    margin: 0.7rem 0.3rem 0 0;
}
.b-indigo { background: rgba(99,102,241,0.2); border: 1px solid rgba(99,102,241,0.5); color: #a5b4fc; }
.b-green  { background: rgba(16,185,129,0.2); border: 1px solid rgba(16,185,129,0.4); color: #6ee7b7; }
.b-pink   { background: rgba(245,101,175,0.2); border: 1px solid rgba(245,101,175,0.4); color: #f9a8d4; }

.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.8rem;
    margin-bottom: 1.4rem;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s, box-shadow 0.3s;
}
.card:hover {
    border-color: rgba(99,102,241,0.35);
    box-shadow: 0 10px 35px rgba(99,102,241,0.12);
}
.card-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.3rem; font-weight: 600; color: #e0e4ff; margin-bottom: 0.3rem; }
.card-sub { font-size: 0.78rem; color: rgba(212,216,240,0.48); margin-bottom: 1.2rem; letter-spacing: 0.03em; }

.result-box {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(99,102,241,0.08));
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 13px; padding: 1.3rem 1.5rem; margin-top: 1rem;
    animation: fadeUp 0.45s ease;
}
.result-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; color: #34d399; margin-bottom: 0.45rem; }
.result-text { font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 500; color: #e8ecff; line-height: 1.5; }

.err-box {
    background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.35);
    border-radius: 12px; padding: 1rem 1.4rem; margin-top: 1rem;
    color: #fca5a5; font-size: 0.86rem;
}
.stat { display: inline-block; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); border-radius: 30px; padding: 0.22rem 0.8rem; font-size: 0.73rem; color: rgba(212,216,240,0.55); margin-top: 0.55rem; }

.sum-col { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; padding: 1.3rem; }
.sum-col-r { border-color: rgba(16,185,129,0.22); }
.col-label { font-size: 0.7rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.7rem; }
.col-label-grey { color: #94a3b8; }
.col-label-green { color: #34d399; }
.col-text { font-size: 0.88rem; line-height: 1.7; color: #c8cce8; }

.divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent); margin: 1.8rem 0; }
.footer { text-align: center; padding: 1.2rem 0 0.8rem; font-size: 0.76rem; color: rgba(212,216,240,0.32); letter-spacing: 0.04em; }
.brand { font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 700; background: linear-gradient(135deg,#a5b4fc,#34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.chip { display:inline-block; background:rgba(99,102,241,0.18); border:1px solid rgba(99,102,241,0.35); border-radius:20px; padding:0.15rem 0.65rem; font-size:0.7rem; color:#a5b4fc !important; margin:0.1rem; font-weight:500; }
.meta { font-size:0.73rem; color:rgba(212,216,240,0.42) !important; line-height:1.65; }

h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; }

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    border-radius: 10px !important;
    color: #d4d8f0 !important;
    font-family: 'Inter',sans-serif !important;
    font-size: 0.94rem !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: rgba(99,102,241,0.65) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.14) !important;
}

.stButton > button {
    background: linear-gradient(135deg,#6366f1,#4f46e5) !important;
    color: #fff !important; font-weight: 600 !important;
    border-radius: 10px !important; padding: 0.52rem 2.1rem !important;
    border: none !important; font-family: 'Inter',sans-serif !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.3) !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#818cf8,#6366f1) !important;
    box-shadow: 0 6px 22px rgba(99,102,241,0.45) !important;
    transform: translateY(-1px) !important;
}

.stRadio > div > label {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important; padding: 0.55rem 1rem !important;
    transition: all 0.2s !important;
}
.stRadio > div > label:hover { border-color: rgba(99,102,241,0.4) !important; background: rgba(99,102,241,0.08) !important; }

@keyframes fadeUp { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
</style>
""", unsafe_allow_html=True)

# ─── Model Loading (per-model, fault-tolerant) ────────────────────────────────
@st.cache_resource
def load_cached_models():
    """Load every model independently so one missing file never blocks the rest."""
    import pickle
    from pathlib import Path
    from tensorflow.keras.models import load_model as keras_load
    from project_paths import resolve_asset

    config = load_config()
    files   = config["files"]
    folders = config["folders"]

    # ── Hindi model ─────────────────────────────────────────────────────────
    hindi_model = tok = None
    try:
        hindi_model = keras_load(str(resolve_asset(files["hindi_model"],     folders["models"])))
        with open(str(resolve_asset(files["hindi_tokenizer"], folders["models"])), "rb") as f:
            tok = pickle.load(f)
    except Exception as e:
        logging.getLogger(__name__).warning("Hindi model unavailable: %s", e)

    # ── Summarizer ──────────────────────────────────────────────────────────
    summarizer_model = sum_tok = None
    try:
        summarizer_model = keras_load(str(resolve_asset(files["summarizer_model"],     folders["models"])))
        with open(str(resolve_asset(files["summarizer_tokenizer"], folders["models"])), "rb") as f:
            sum_tok = pickle.load(f)
    except Exception as e:
        logging.getLogger(__name__).warning("Summarizer model unavailable: %s", e)

    # ── Spanish (HuggingFace) ───────────────────────────────────────────────
    spanish_tokenizer = spanish_model = None
    try:
        from transformers import MarianTokenizer, MarianMTModel
        spanish_pretrained = config.get("spanish_pretrained", "Helsinki-NLP/opus-mt-en-es")
        spanish_tokenizer = MarianTokenizer.from_pretrained(spanish_pretrained)
        spanish_model     = MarianMTModel.from_pretrained(spanish_pretrained)
    except Exception as e:
        logging.getLogger(__name__).warning("Spanish model unavailable: %s", e)

    # ── Hindi HuggingFace fallback (used when .h5 file is missing) ──────────
    hindi_hf_tokenizer = hindi_hf_model = None
    if hindi_model is None:
        try:
            from transformers import MarianTokenizer, MarianMTModel
            logging.getLogger(__name__).info("Loading HuggingFace Hindi fallback model...")
            hindi_hf_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
            hindi_hf_model     = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-hi")
            logging.getLogger(__name__).info("Hindi HuggingFace model loaded.")
        except Exception as e:
            logging.getLogger(__name__).warning("Hindi HuggingFace fallback unavailable: %s", e)

    return hindi_model, tok, summarizer_model, sum_tok, spanish_tokenizer, spanish_model, hindi_hf_tokenizer, hindi_hf_model

hindi_model, tok, summarizer_model, sum_tok, spanish_tokenizer, spanish_model, hindi_hf_tokenizer, hindi_hf_model = load_cached_models()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="brand">🌐 LinguaVerse AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="meta">Neural Language Studio</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Select Mode**")
    mode = st.radio("", [
        "🇮🇳 English → Hindi",
        "🇪🇸 English → Spanish",
        "📝 Summarizer"
    ], label_visibility="collapsed")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="meta">
        <span class="chip">TensorFlow</span>
        <span class="chip">Keras</span>
        <span class="chip">HuggingFace</span>
        <br><br>
        <strong style="color:#a5b4fc">I050 · Yug Nagda</strong><br>
        ATML Lab 6 · NMIMS B.Tech AI
    </div>
    """, unsafe_allow_html=True)

# ─── Hero Banner ─────────────────────────────────────────────────────────────
_meta = {
    "🇮🇳 English → Hindi":   ("🇮🇳 English → Hindi Translation",  "Encoder-Decoder LSTM",   "b-indigo"),
    "🇪🇸 English → Spanish": ("🇪🇸 English → Spanish Translation","HuggingFace MarianMT",   "b-green"),
    "📝 Summarizer":          ("📝 Neural Text Summarizer",         "Custom Encoder-Decoder", "b-pink"),
}
t, m, bc = _meta[mode]
st.markdown(f"""
<div class="hero-banner">
    <div class="hero-title">{t}</div>
    <div class="hero-sub">{m} · Neural Language Processing</div>
    <span class="badge b-indigo">I050</span>
    <span class="badge b-green">Yug Nagda</span>
    <span class="badge {bc}">{m}</span>
</div>
""", unsafe_allow_html=True)

# ─── Mode 1: Hindi ───────────────────────────────────────────────────────────
if "Hindi" in mode:
    # Choose caption based on which backend is active
    _hin_caption = (
        "Encoder-Decoder LSTM trained 50 epochs on Hindi-English parallel corpus"
        if hindi_model is not None
        else "Helsinki-NLP/opus-mt-en-hi · HuggingFace MarianMT (neural machine translation)"
    )
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Enter English Text</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card-sub">{_hin_caption}</div>', unsafe_allow_html=True)
    text = st.text_input("", placeholder="e.g. How are you?", key="h_in", label_visibility="collapsed")
    if st.button("Translate →", key="hin"):
        if text.strip():
            with st.spinner("Translating..."):
                try:
                    if hindi_model is not None:
                        # Use trained Keras model
                        result = translate_to_hindi(text, hindi_model, tok)
                    elif hindi_hf_model is not None:
                        # Use HuggingFace MarianMT fallback
                        inputs = hindi_hf_tokenizer([text], return_tensors="pt", padding=True)
                        translated = hindi_hf_model.generate(**inputs)
                        result = hindi_hf_tokenizer.decode(translated[0], skip_special_tokens=True)
                    else:
                        raise RuntimeError("Hindi translation model is not available. Please check your internet connection and restart the app.")
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">🇮🇳 Hindi Translation</div>
                        <div class="result-text">{result}</div>
                    </div>
                    <span class="stat">📥 {len(text.split())} words input</span>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="err-box">⚠️ {e}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter some English text.")
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Mode 2: Spanish ─────────────────────────────────────────────────────────
elif "Spanish" in mode:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Enter English Text</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">Helsinki-NLP/opus-mt-en-es · HuggingFace MarianMT pre-trained model</div>', unsafe_allow_html=True)
    text = st.text_input("", placeholder="e.g. The weather is beautiful today.", key="s_in", label_visibility="collapsed")
    if st.button("Translate →", key="esp"):
        if text.strip():
            with st.spinner("Translating..."):
                try:
                    if spanish_tokenizer is None or spanish_model is None:
                        raise RuntimeError("Spanish model unavailable — may require internet access to download.")
                    result = translate_to_spanish(text, spanish_tokenizer, spanish_model)
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">🇪🇸 Spanish Translation</div>
                        <div class="result-text">{result}</div>
                    </div>
                    <span class="stat">📥 {len(text.split())} words input</span>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="err-box">⚠️ {e}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter some English text.")
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Mode 3: Summarizer ──────────────────────────────────────────────────────
elif "Summ" in mode:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Paste Your Text</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-sub">Custom Encoder-Decoder trained on news corpus · Distills content while preserving meaning</div>', unsafe_allow_html=True)
    text = st.text_area("", height=200, placeholder="Paste a paragraph or article here...", key="sum_in", label_visibility="collapsed")
    if st.button("Summarize →", key="sum"):
        if text.strip():
            with st.spinner("Summarizing..."):
                try:
                    summary = summarize_text(text, summarizer_model, sum_tok)
                    orig_w = len(text.split())
                    summ_w = len(summary.split())
                    reduction = round((1 - summ_w / orig_w) * 100) if orig_w > 0 else 0
                    col1, col2 = st.columns(2, gap="medium")
                    with col1:
                        st.markdown(f"""
                        <div class="sum-col">
                            <div class="col-label col-label-grey">📄 Original</div>
                            <div class="col-text">{text}</div>
                        </div>
                        <span class="stat">📝 {orig_w} words</span>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown(f"""
                        <div class="sum-col sum-col-r">
                            <div class="col-label col-label-green">✨ Summary</div>
                            <div class="col-text">{summary}</div>
                        </div>
                        <span class="stat">✂️ {summ_w} words · 📉 {reduction}% shorter</span>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.markdown(f'<div class="err-box">⚠️ {e}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to summarize.")
    st.markdown('</div>', unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    🌐 LinguaVerse AI &nbsp;·&nbsp; Streamlit + TensorFlow + HuggingFace
    &nbsp;·&nbsp; <strong style="color:#a5b4fc">I050 · Yug Nagda</strong>
    &nbsp;·&nbsp; NMIMS B.Tech AI · ATML Lab 6
</div>
""", unsafe_allow_html=True)
