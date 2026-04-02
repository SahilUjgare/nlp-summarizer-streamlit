# 📝 NLP Text Summarizer — T5 + Streamlit

A production-grade, deep learning-powered **Text Summarization App** built with:
- 🤖 **Google T5** (via HuggingFace Transformers)
- 🔥 **PyTorch** (No TensorFlow required)
- 🌊 **Streamlit** UI

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Deploy on Streamlit Cloud

1. Push this folder to a GitHub repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entry point
4. Deploy ✅

---

## 🧠 How It Works

| Component       | Detail                        |
|-----------------|-------------------------------|
| Model           | `t5-small` / `t5-base`        |
| Task prefix     | `"summarize: "`               |
| Decoding        | Beam Search (configurable)    |
| Framework       | PyTorch + HuggingFace         |
| No TensorFlow   | ✅ Pure PyTorch                |

---

## ⚙️ Features

- ✅ Real DL model (T5 Transformer)
- ✅ Adjustable beam width, min/max length, length penalty
- ✅ Live word/char/sentence stats
- ✅ Compression ratio & inference time metrics
- ✅ 3 built-in sample texts
- ✅ Download summary as `.txt`
- ✅ Industry-level dark UI

  
