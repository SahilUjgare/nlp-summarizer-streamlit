import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import time
import re

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Summarizer · T5",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

/* Background */
.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* Hide default header */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f0f1a;
    border-right: 1px solid #1e1e2e;
}

section[data-testid="stSidebar"] * {
    color: #c0c0d8 !important;
}

/* Hero Title */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.2rem;
    line-height: 1.1;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #6366f1;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

/* Cards */
.metric-card {
    background: linear-gradient(145deg, #12121f, #1a1a2e);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    transition: border-color 0.3s;
}

.metric-card:hover {
    border-color: #6366f1;
}

.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #a78bfa;
}

.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Input area */
.stTextArea textarea {
    background: #0f0f1a !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 10px !important;
    color: #e8e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
}

.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.15) !important;
}

/* Sliders */
.stSlider > div > div > div > div {
    background: #6366f1 !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 0.05em !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
}

/* Summary output box */
.summary-box {
    background: linear-gradient(145deg, #0d1117, #111827);
    border: 1px solid #1e3a5f;
    border-left: 4px solid #60a5fa;
    border-radius: 12px;
    padding: 1.8rem;
    margin-top: 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.92rem;
    line-height: 1.8;
    color: #c7d2fe;
}

.summary-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #60a5fa;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Model badge */
.model-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.4);
    color: #a78bfa;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    margin-bottom: 1rem;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #1e1e2e;
    margin: 2rem 0;
}

/* Success / Info boxes */
.stSuccess, .stInfo {
    background: #0f1a0f !important;
    border-color: #34d399 !important;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #8b5cf6, #a78bfa) !important;
}

/* Select box */
.stSelectbox select {
    background: #0f0f1a !important;
    color: #e8e8f0 !important;
    border-color: #2a2a4a !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2a4a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }
</style>
""", unsafe_allow_html=True)


# ─── Model Loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


# ─── Summarization Logic ───────────────────────────────────────────────────────
def summarize(text: str, tokenizer, model, min_len: int, max_len: int, num_beams: int, length_penalty: float) -> str:
    prefix = "summarize: "
    input_ids = tokenizer.encode(
        prefix + text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    )
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            min_length=min_len,
            max_length=max_len,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def count_words(text: str) -> int:
    return len(re.findall(r'\w+', text))


def compute_compression(original: str, summary: str) -> float:
    orig_w = count_words(original)
    summ_w = count_words(summary)
    if orig_w == 0:
        return 0.0
    return round((1 - summ_w / orig_w) * 100, 1)


# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Model Config")

    model_choice = st.selectbox(
        "Model",
        ["t5-small", "t5-base"],
        index=0,
        help="t5-small is fastest; t5-base is more accurate"
    )

    st.markdown("---")
    st.markdown("### 📐 Summary Length")

    min_length = st.slider("Min tokens", 30, 100, 50, step=5)
    max_length = st.slider("Max tokens", 80, 300, 150, step=10)

    st.markdown("---")
    st.markdown("### 🔬 Decoding Params")

    num_beams = st.slider("Beam width", 1, 8, 4, step=1,
                          help="Higher = better quality, slower")
    length_penalty = st.slider("Length penalty", 0.5, 2.0, 1.2, step=0.1,
                               help=">1 encourages longer summaries")

    st.markdown("---")
    st.markdown("""
    <div style='font-family: JetBrains Mono, monospace; font-size: 0.72rem; color: #4b5563; line-height: 1.8;'>
    🧠 Model: HuggingFace T5<br>
    ⚡ Backend: PyTorch<br>
    📦 No TensorFlow required<br>
    🚀 Deployable on Streamlit Cloud
    </div>
    """, unsafe_allow_html=True)

# ─── Main Content ──────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">NLP Text Summarizer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">⚡ Powered by Google T5 · Deep Learning · Transformers</div>', unsafe_allow_html=True)
st.markdown(f'<span class="model-badge">🤖 {model_choice} · Beam Search · PyTorch</span>', unsafe_allow_html=True)

# Load model
with st.spinner(f"Loading {model_choice} model..."):
    tokenizer, model = load_model(model_choice)

st.success(f"✅ `{model_choice}` loaded and ready!")

st.markdown("---")

# ─── Input ────────────────────────────────────────────────────────────────────
col_input, col_info = st.columns([3, 1])

with col_input:
    st.markdown("#### 📄 Input Text")
    input_text = st.text_area(
        label="Paste your article, research paper, or any long text below:",
        height=280,
        placeholder="Paste your article, research paper, or any long text here...\n\nMinimum ~80 words recommended for best results.",
        label_visibility="collapsed"
    )

with col_info:
    st.markdown("#### 📊 Text Stats")
    word_count = count_words(input_text) if input_text.strip() else 0
    char_count = len(input_text)
    sentence_count = len(re.split(r'[.!?]+', input_text.strip())) if input_text.strip() else 0

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{word_count}</div>
        <div class="metric-label">Words</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{char_count}</div>
        <div class="metric-label">Characters</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{max(0, sentence_count - 1)}</div>
        <div class="metric-label">Sentences</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("")

# ─── Example texts ────────────────────────────────────────────────────────────
with st.expander("💡 Load a sample text"):
    sample_texts = {
        "AI & Deep Learning": """Deep learning is a subset of machine learning that uses neural networks with many layers to model complex patterns in data. These networks, inspired by the human brain, automatically learn representations from raw data such as images, text, and audio. The development of deep learning has been driven by the availability of large datasets, advances in computational power particularly GPUs, and improvements in training algorithms. Applications of deep learning include image recognition, natural language processing, speech recognition, and drug discovery. Major breakthroughs include convolutional neural networks for image tasks, recurrent neural networks for sequential data, and transformers for natural language understanding. Companies like Google, Meta, and OpenAI have invested heavily in deep learning research, producing models like BERT, GPT, and AlphaFold that have reshaped their respective fields.""",

        "Climate Change": """Climate change refers to long-term shifts in temperatures and weather patterns on Earth. Since the 1800s, human activities have been the main driver of climate change, primarily due to burning fossil fuels like coal, oil, and gas. Burning fossil fuels generates greenhouse gas emissions that act like a blanket wrapped around the Earth, trapping the sun's heat and raising temperatures. The consequences include rising sea levels, more frequent extreme weather events, disruption of ecosystems, and threats to food and water security. International agreements like the Paris Agreement aim to limit global warming to 1.5°C above pre-industrial levels. Renewable energy, energy efficiency, sustainable agriculture, and reforestation are among the key strategies to mitigate climate change and its effects on human civilization and biodiversity.""",

        "Quantum Computing": """Quantum computing leverages the principles of quantum mechanics to perform computations that are infeasible for classical computers. Unlike classical bits that represent either 0 or 1, quantum bits or qubits can exist in superpositions of both states simultaneously, enabling massive parallelism. Quantum entanglement allows qubits to be correlated in ways that have no classical equivalent, while quantum interference helps amplify correct solutions and cancel errors. Quantum computers promise to revolutionize fields such as cryptography, drug discovery, optimization, and materials science. Companies including IBM, Google, and startups like IonQ and Rigetti are racing to build practical quantum systems. Google claimed quantum supremacy in 2019 when their Sycamore processor completed a specific calculation in 200 seconds that would have taken classical supercomputers thousands of years.""",
    }

    chosen_sample = st.selectbox("Choose a sample:", list(sample_texts.keys()))
    if st.button("📋 Load Sample"):
        st.session_state["sample_loaded"] = sample_texts[chosen_sample]
        st.rerun()

if "sample_loaded" in st.session_state and not input_text.strip():
    input_text = st.session_state["sample_loaded"]

# ─── Summarize Button ──────────────────────────────────────────────────────────
st.markdown("")
run_btn = st.button("🚀 Summarize Now", use_container_width=True)

# ─── Output ───────────────────────────────────────────────────────────────────
if run_btn:
    if not input_text.strip():
        st.warning("⚠️ Please enter some text first.")
    elif word_count < 30:
        st.warning("⚠️ Text is too short. Please enter at least 30 words for meaningful summarization.")
    else:
        with st.spinner("🧠 Running T5 inference..."):
            progress_bar = st.progress(0)
            for i in range(1, 60):
                time.sleep(0.015)
                progress_bar.progress(i)

            start = time.time()
            summary = summarize(input_text, tokenizer, model, min_length, max_length, num_beams, length_penalty)
            elapsed = round(time.time() - start, 2)

            for i in range(60, 101):
                time.sleep(0.008)
                progress_bar.progress(i)

        compression = compute_compression(input_text, summary)
        summary_words = count_words(summary)

        # Metrics row
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{word_count}</div><div class="metric-label">Original Words</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{summary_words}</div><div class="metric-label">Summary Words</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{compression}%</div><div class="metric-label">Compression</div></div>', unsafe_allow_html=True)
        with m4:
            st.markdown(f'<div class="metric-card"><div class="metric-value">{elapsed}s</div><div class="metric-label">Inference Time</div></div>', unsafe_allow_html=True)

        # Summary box
        st.markdown(f"""
        <div class="summary-box">
            <div class="summary-header">✦ Generated Summary · {model_choice} · {num_beams}-beam search</div>
            {summary}
        </div>
        """, unsafe_allow_html=True)

        # Download
        st.markdown("")
        st.download_button(
            label="⬇️ Download Summary as .txt",
            data=f"ORIGINAL TEXT:\n{input_text}\n\n{'='*60}\n\nSUMMARY:\n{summary}\n\nModel: {model_choice} | Words: {word_count}→{summary_words} | Compression: {compression}%",
            file_name="summary_output.txt",
            mime="text/plain"
        )

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align: center; font-family: JetBrains Mono, monospace; font-size: 0.72rem; color: #374151; padding-bottom: 1.5rem;'>
Built with 🤖 HuggingFace Transformers · 🔥 PyTorch · 🌊 Streamlit · Google T5 Architecture
</div>
""", unsafe_allow_html=True)
