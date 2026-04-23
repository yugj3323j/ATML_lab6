# 🌐 LinguaVerse AI — ATML Lab 6

### Neural Language Studio · Encoder-Decoder with Attention

> **Author:** Yug Nagda &nbsp;|&nbsp; **ID:** I050 &nbsp;|&nbsp; NMIMS B.Tech AI — ATML Lab 6

---

## 🚀 Overview

**LinguaVerse AI** is a full-stack neural language processing application built with a custom **Encoder-Decoder LSTM** architecture (with Attention) for sequence-to-sequence tasks. It ships with an interactive **Streamlit dashboard** featuring a premium glassmorphism dark-mode UI.

### ✨ Capabilities

| Mode | Description | Backend |
|------|-------------|---------|
| 🇮🇳 English → Hindi | Seq2Seq LSTM translation | Custom Keras model (50 epochs) |
| 🇪🇸 English → Spanish | Pre-trained MarianMT | HuggingFace `opus-mt-en-es` |
| 📝 Text Summarizer | Abstractive summarization | Custom Encoder-Decoder (Keras) |

---

## 📁 Project Structure

```
Encoder-DecoderArchitecture-main/
│
├── I050_Yug Nagda_ATML_Lab 6/
│   ├── app.py                        # ✨ Streamlit frontend (LinguaVerse AI)
│   ├── main.py                       # Pipeline entry point
│   ├── pipeline.py                   # End-to-end pipeline runner
│   ├── inference.py                  # Translation & summarization logic
│   ├── model_loader.py               # Model loading utilities
│   ├── project_paths.py              # Dynamic path resolution
│   ├── config.json                   # Configuration file
│   ├── custom_summarizer_model.keras # Trained summarizer weights
│   ├── summarizer_tokenizer_data.pkl # Summarizer tokenizer
│   ├── tokenizer_data.pkl            # Hindi translation tokenizer
│   ├── input/                        # Input data directory
│   ├── outputs/                      # Generated outputs
│   └── requirements.txt              # Python dependencies
│
└── README.md
```

---



# Install dependencies
pip install -r "I050_Yug Nagda_ATML_Lab 6/requirements.txt"
```

---

## ▶️ How to Run

### 🌐 Launch the Streamlit Dashboard

```bash
cd "I050_Yug Nagda_ATML_Lab 6"
streamlit run app.py
```

Open your browser at:

```
http://localhost:8501
```

---

### ⚡ Run Full Pipeline

```bash
cd "I050_Yug Nagda_ATML_Lab 6"
python main.py
```

This executes the full pipeline:
- ✅ Data loading & preprocessing
- ✅ Model inference
- ✅ Output generation (saved to `outputs/`)

---

## 🎨 UI Highlights

The Streamlit dashboard features a **premium glassmorphism dark-mode design**:

- 🎨 Radial dark gradient background with floating orbs
- ✨ Animated shimmer header bar (indigo → emerald → pink)
- 🃏 Frosted-glass mode cards with hover lift
- 🔤 Space Grotesk + Inter typography (Google Fonts)
- 💡 Animated result boxes with fade-up effect
- 🏷️ Badge chips for model metadata
- 📊 Word count & compression stats for summarizer

---

## 🔄 Features

### ✅ Dynamic Path Handling
- No hardcoded file paths — uses `pathlib` and `config.json`
- Cross-platform compatible (Windows, macOS, Linux)

### ✅ Cached Model Loading
- `@st.cache_resource` prevents redundant loads
- All three models loaded once at startup

### ✅ Robust Error Handling
- Per-mode try/except with styled error boxes
- Graceful fallback if Spanish model is offline

### ✅ Modular Architecture
- Separate `inference.py`, `model_loader.py`, `pipeline.py`
- Clean separation of concerns

---

## 🛠️ Dependencies

```
streamlit
tensorflow
transformers
torch
sentencepiece
numpy
```

> Full list in `requirements.txt`

---

## ❗ Troubleshooting

### Streamlit not found
```bash
pip install streamlit
```

### Port already in use
```bash
streamlit run app.py --server.port 8502
```

### Spanish model requires internet
The HuggingFace MarianMT model (`Helsinki-NLP/opus-mt-en-es`) is downloaded on first use. Ensure internet access or pre-cache it.

### Missing model files
Ensure the following files exist in the project directory:
- `custom_summarizer_model.keras`
- `summarizer_tokenizer_data.pkl`
- `tokenizer_data.pkl`

---

## 💡 Future Improvements

- [ ] Add English → French / German translation modes
- [ ] Integrate BLEU score evaluation per translation
- [ ] Add attention heatmap visualization
- [ ] Dockerize the entire application
- [ ] Add model fine-tuning UI in Streamlit
- [ ] Export translations/summaries as PDF

---

## 👨‍💻 Author

**Yug Nagda**
Roll No: **I050**
NMIMS B.Tech Artificial Intelligence
ATML Lab 6 — Encoder-Decoder with Attention
