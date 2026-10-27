"""Plant Disease Detection - Streamlit app.

Run:  streamlit run app.py
Needs: model.h5 in the same folder as this file (or downloads automatically from Google Drive).
"""
import base64
import io
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import gdown
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from disease_info import CLASS_NAMES, DISEASE_INFO, SUPPORTED_CROPS

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.h5"
IMG_SIZE = (224, 224)  # model input: (None, 224, 224, 3)

PREPROCESS_MODES = {
    "Divide by 255  (0 to 1)": "div255",
    "Xception style  (-1 to 1)": "xception",
    "Raw pixels  (0 to 255)": "raw",
}

ADVISOR_PROMPT = (
    "You are a careful plant-health assistant for home gardeners and small farmers growing "
    "tomato and potato. Answer briefly in plain language. Never give pesticide dosages; tell the "
    "user to follow the product label and local regulations. If unsure, recommend contacting the "
    "local agricultural extension service."
)

st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="centered",
                    initial_sidebar_state="collapsed")


# --------------------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading AI model (downloading if needed)...")
def load_model():
    import tensorflow as tf  # imported lazily so the page appears instantly

    # Automatically download model from Google Drive if missing (bypasses GitHub 25MB limit)
    if not MODEL_PATH.exists():
        st.info("📥 Downloading model from Google Drive (first-time cloud boot, ~89MB)... Please wait.")
        gdrive_url = "https://drive.google.com/uc?id=1wuHFBZs-76upHrPybchq4LZbh8khOTAP&export=download"
        gdown.download(gdrive_url, str(MODEL_PATH), quiet=False)

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def preprocess(img: Image.Image, mode: str) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype="float32")
    if mode == "div255":
        arr = arr / 255.0
    elif mode == "xception":
        arr = arr / 127.5 - 1.0
    return np.expand_dims(arr, 0)


@st.cache_data(show_spinner=False, max_entries=32)
def predict(image_bytes: bytes, mode: str) -> list:
    model = load_model()
    img = Image.open(io.BytesIO(image_bytes))
    probs = model.predict(preprocess(img, mode), verbose=0)[0]
    return [float(p) for p in probs]


# ------------------------------------------------------------------ helpers
def pretty(class_name: str) -> str:
    crop, _, cond = class_name.partition("___")
    crop = crop.replace("_", " ").replace(",", "")
    cond = cond.replace("_", " ").strip()
    return f"{crop} - {cond}"


def crop_of(class_name: str) -> str:
    return class_name.split("___")[0].split("_")[0].split(",")[0]


def ask_ollama(question: str, context: str, model_name: str, host: str) -> str:
    """Free, open-source LLM running locally via Ollama (https://ollama.com)."""
    payload = json.dumps({
        "model": model_name,
        "stream": False,
        "messages": [
            {"role": "system", "content": ADVISOR_PROMPT},
            {"role": "user", "content": f"Diagnosis from image model: {context}\n\nQuestion: {question}"},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(f"{host.rstrip('/')}/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())["message"]["content"]


def bullets(items):
    for item in items:
        st.markdown(f"- {item}")



# ------------------------------------------------------------------ styling
CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 820px;}
.hero {background: linear-gradient(135deg, #1b5e20 0%, #43a047 100%); color: #fff;
       border-radius: 20px; padding: 28px 24px 22px; text-align: center; margin-bottom: 18px;}
.hero h1 {color: #fff; margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: .3px;}
.hero p {color: #e8f5e9; margin: 6px 0 0; font-size: 1.02rem;}
.crops {display: flex; gap: 14px; margin-bottom: 20px;}
.crop {flex: 1; border-radius: 16px; padding: 14px 10px 12px; text-align: center;
       background: rgba(76,175,80,.10); border: 1px solid rgba(76,175,80,.35);}
.crop svg, .crop img {width: 96px; height: 96px; object-fit: cover; border-radius: 14px;}
.crop b {display: block; margin-top: 6px; font-size: 1.05rem;}
.crop span {font-size: .85rem; opacity: .75;}
.banner {border-radius: 14px; padding: 14px 18px; margin: 10px 0 14px; color: #fff;
         font-weight: 700; font-size: 1.1rem;}
.b-none {background: #2e7d32;} .b-moderate {background: #ef8f00;}
.b-high {background: #d32f2f;} .b-incurable {background: #7f1d1d;}
.card {border-radius: 14px; padding: 14px 18px; margin: 8px 0;
       background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.2);}
.stat {text-align: center; border-radius: 14px; padding: 12px;
       background: rgba(76,175,80,.10); border: 1px solid rgba(76,175,80,.3);}
.stat small {opacity: .7; display: block;} .stat b {font-size: 1.25rem;}
.foot {text-align: center; opacity: .6; font-size: .8rem; margin-top: 28px;}
</style>
"""

TOMATO_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
<circle cx="50" cy="56" r="34" fill="#e53935"/><ellipse cx="38" cy="42" rx="9" ry="5" fill="#ff8a80" opacity=".6" transform="rotate(-30 38 42)"/>
<path d="M50 24 L43 14 L50 19 L57 14 Z M50 24 L33 24 L44 30 Z M50 24 L67 24 L56 30 Z M50 24 L38 34 L48 30 Z M50 24 L62 34 L52 30 Z"
 fill="#2e7d32"/><rect x="48" y="10" width="4" height="12" rx="2" fill="#1b5e20"/></svg>"""

POTATO_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
<path d="M18 58 C14 36 34 22 56 26 C78 30 90 46 82 66 C74 84 40 88 26 76 C21 72 19 66 18 58 Z" fill="#c8923f"/>
<path d="M30 40 C40 30 58 30 70 38" stroke="#e6b868" stroke-width="4" fill="none" stroke-linecap="round" opacity=".7"/>
<circle cx="40" cy="58" r="2.6" fill="#8d5a1e"/><circle cx="58" cy="52" r="2.6" fill="#8d5a1e"/>
<circle cx="66" cy="68" r="2.6" fill="#8d5a1e"/><circle cx="46" cy="72" r="2.6" fill="#8d5a1e"/>
<circle cx="30" cy="62" r="2.2" fill="#8d5a1e"/></svg>"""


def crop_visual(name: str, svg: str) -> str:
    """Use your own photo from images/<name>.jpg|png|webp if present, else the built-in drawing."""
    for ext in ("jpg", "jpeg", "png", "webp"):
        p = BASE_DIR / "images" / f"{name}.{ext}"
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'<img src="data:image/{ext};base64,{b64}">'
    return svg


def hero_and_crops():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><h1>🌿 Plant Disease Detection</h1>'
        '<p>Upload a leaf photo and get an instant diagnosis with treatment advice</p></div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="crops">'
        f'<div class="crop">{crop_visual("tomato", TOMATO_SVG)}<b>Tomato</b><span>9 diseases + healthy</span></div>'
        f'<div class="crop">{crop_visual("potato", POTATO_SVG)}<b>Potato</b><span>2 diseases + healthy</span></div>'
        f'</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Minimum confidence", 40, 95, 70, 5, format="%d%%") / 100
    with st.expander("Advanced"):
        mode_label = st.selectbox(
            "Image scaling used in training", list(PREPROCESS_MODES),
            help="Must match how the model was trained. If healthy leaves look wrong, try another option.")
    use_llm = st.toggle("AI advisor (Ollama)", value=False,
                        help="Optional. Needs Ollama installed and a model pulled, e.g. 'ollama pull llama3.2'.")
    if use_llm:
        llm_model = st.text_input("Ollama model", "llama3.2")
        llm_host = st.text_input("Ollama address", "http://localhost:11434")

mode = PREPROCESS_MODES[mode_label]

# ------------------------------------------------------------------ main page
hero_and_crops()

tab_up, tab_cam = st.tabs(["📁 Upload photo", "📷 Use camera"])
with tab_up:
    uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png", "webp"],
                                label_visibility="collapsed")
with tab_cam:
    camera = st.camera_input("Take a photo of the leaf", label_visibility="collapsed")

source = uploaded or camera
if source is None:
    st.info("💡 Tip: use a close, well-lit photo of ONE leaf on a plain background.")
    st.markdown('<div class="foot">Decision aid only - confirm serious outbreaks with a local '
                'agricultural extension service.</div>', unsafe_allow_html=True)
    st.stop()

image_bytes = source.getvalue()
try:
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
except Exception:
    st.error("That file could not be read as an image. Please try another photo.")
    st.stop()

try:
    with st.spinner("Analysing leaf..."):
        probs = predict(image_bytes, mode)
except Exception as exc:
    st.error("The model could not run. This is usually a TensorFlow/Keras version mismatch.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.info("Fix: `pip install -U \"tensorflow>=2.18\"`, then restart the app.")
    st.stop()

if len(probs) != len(CLASS_NAMES):
    st.error(f"Model has {len(probs)} outputs but the app expects {len(CLASS_NAMES)} classes. "
             "Update CLASS_NAMES in disease_info.py to match your training order.")
    st.stop()

order = np.argsort(probs)[::-1]
best = int(order[0])
best_name, conf = CLASS_NAMES[best], probs[best]
info = DISEASE_INFO.get(best_name)

col_img, col_res = st.columns([1, 1.2], gap="large")
with col_img:
    st.image(image, caption="Your leaf")
with col_res:
    st.markdown(f'<div class="stat"><small>Prediction</small><b>{pretty(best_name)}</b></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="stat" style="margin-top:10px"><small>Confidence</small>'
                f'<b>{conf:.1%}</b></div>', unsafe_allow_html=True)
    st.progress(min(max(conf, 0.0), 1.0))

top = pd.DataFrame({"Class": [pretty(CLASS_NAMES[i]) for i in order[:3]],
                    "Probability": [probs[i] for i in order[:3]]}).set_index("Class")
with st.expander("Top 3 possibilities"):
    st.bar_chart(top, horizontal=True)

if crop_of(best_name) not in SUPPORTED_CROPS:
    st.warning(f"This looks like **{pretty(best_name)}**, which this app does not cover. "
               "Please upload a tomato or potato leaf.")
    st.stop()
if conf < threshold:
    st.warning("The model is **not confident** about this photo. Retake it closer, in daylight, "
               "with one leaf filling most of the frame, and avoid blur or heavy shadows.")
    st.stop()

# -------- diagnosis
severity = info["severity"]
label = {"none": "✅ Healthy leaf - no treatment needed",
         "moderate": f"⚠️ {info['title']} - treat promptly",
         "high": f"🚨 {info['title']} - spreads fast, act today",
         "incurable": f"🛑 {info['title']} - no cure, protect the rest of the crop"}[severity]
st.markdown(f'<div class="banner b-{severity}">{label}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="card"><b>What it is</b><br>{info["cause"]}</div>', unsafe_allow_html=True)

t_sym, t_treat, t_prev = st.tabs(["🔍 Symptoms", "💊 Treatment & Remedies", "🛡️ Prevention"])
with t_sym:
    bullets(info["symptoms"])
with t_treat:
    st.markdown("🌿 **Home / Organic Remedies**")
    bullets(info["cultural"])
    if info["chemical"]:
        st.markdown("💊 **Commercial Treatment & Dosages**")
        bullets(info["chemical"])
        st.markdown("🛒 **Local Store / Online Search Tip:** Search the medicine name above on agricultural supply stores or delivery apps (like Coupang or local garden supply centers) to find available brands easily.")
        st.caption("Always follow the product label for exact safety precautions, mixing ratios, and local agricultural regulations.")
with t_prev:
    bullets(info["prevention"])

# -------- optional AI advisor
if use_llm:
    st.divider()
    st.subheader("💬 Ask the AI advisor")
    q = st.text_input("Your question", placeholder="e.g. Can I still eat the tomatoes?")
    if st.button("Ask") and q.strip():
        try:
            with st.spinner("Thinking..."):
                st.markdown(ask_ollama(q, f"{info['title']} ({conf:.0%} confidence)", llm_model, llm_host))
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            st.error(f"Could not reach Ollama at {llm_host}. Is it running, and is the model pulled?\n\n{exc}")

st.markdown('<div class="foot">Decision aid only - confirm serious outbreaks with a local '
            'agricultural extension service.</div>', unsafe_allow_html=True)
