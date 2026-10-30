"""Plant Disease Detection - password-protected, responsive Streamlit app.

Run:  streamlit run app.py
Needs: disease_info.py next to this file, and model.h5 (auto-downloads from Google Drive if missing).
"""
import base64
import hmac
import io
import json
import time
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
IMG_SIZE = (224, 224)

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

# --------------------------------------------------------------------------- styling
CSS = """
<style>
/* ---- hide Streamlit chrome: fork button, GitHub icon, menu, footer, hosted badge ---- */
#MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stToolbar"],
[data-testid="stDeployButton"], [data-testid="stAppDeployButton"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
.stDeployButton, .viewerBadge_container__r5tak, [class*="viewerBadge"],
a[href*="github.com"][target="_blank"] img {display: none !important; visibility: hidden !important;}
header[data-testid="stHeader"] {background: transparent !important;}

/* ---- layout ---- */
.block-container {padding: 1.2rem 1rem 3rem; max-width: 880px;}
@keyframes flow {0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%}}
@keyframes floaty {0%,100%{transform:translateY(0) rotate(-6deg)} 50%{transform:translateY(-10px) rotate(6deg)}}
@keyframes pop {from{opacity:0; transform:translateY(14px)} to{opacity:1; transform:none}}
@keyframes grow {from{width:0}}

/* ---- hero ---- */
.hero {position: relative; overflow: hidden; border-radius: 26px; padding: 34px 22px 28px;
       text-align: center; color: #fff; margin-bottom: 18px;
       background: linear-gradient(120deg,#0b3d1e,#1b8a3a,#7cb342,#1b8a3a,#0b3d1e);
       background-size: 300% 300%; animation: flow 14s ease infinite;
       box-shadow: 0 14px 40px rgba(27,138,58,.35);}
.hero h1 {color:#fff; margin:0; font-size: clamp(1.6rem, 5.5vw, 2.5rem); font-weight: 800; letter-spacing:.3px;}
.hero p {color:#eaf7ea; margin:8px auto 0; max-width: 560px; font-size: clamp(.9rem, 2.8vw, 1.05rem);}
.hero .leaf {position:absolute; font-size: 2.4rem; opacity:.22; animation: floaty 6s ease-in-out infinite;}
.hero .l1 {left: 5%; top: 14%;} .hero .l2 {right: 6%; top: 40%; animation-delay: 1.5s; font-size:3rem;}
.hero .l3 {left: 20%; bottom: 6%; animation-delay: 3s; font-size:1.8rem;}

/* ---- steps ---- */
.steps {display:grid; grid-template-columns: repeat(3,1fr); gap:10px; margin-bottom:18px;}
.step {border-radius:16px; padding:12px 8px; text-align:center; font-size:.85rem;
       background: rgba(76,175,80,.10); border:1px solid rgba(76,175,80,.30); animation: pop .6s both;}
.step b {display:block; font-size:1.5rem;}
.step:nth-child(2){animation-delay:.12s} .step:nth-child(3){animation-delay:.24s}

/* ---- crop cards ---- */
.crops {display:grid; grid-template-columns: 1fr 1fr; gap:14px; margin-bottom:20px;}
.crop {border-radius:20px; padding:16px 10px 14px; text-align:center; transition: transform .25s, box-shadow .25s;
       background: linear-gradient(160deg, rgba(76,175,80,.16), rgba(76,175,80,.04));
       border:1px solid rgba(76,175,80,.35); backdrop-filter: blur(6px);}
.crop:hover {transform: translateY(-4px); box-shadow: 0 10px 26px rgba(76,175,80,.25);}
.crop svg, .crop img {width: min(96px, 24vw); height: min(96px, 24vw); object-fit: cover; border-radius: 16px;}
.crop b {display:block; margin-top:8px; font-size:1.05rem;}
.crop span {font-size:.82rem; opacity:.75;}

/* ---- result cards ---- */
.card {border-radius:16px; padding:14px 18px; margin:8px 0; animation: pop .5s both;
       background: rgba(128,128,128,.08); border:1px solid rgba(128,128,128,.2);}

/* ---- verdict card ---- */
.verdict {display:flex; align-items:center; gap:16px; border-radius:22px; padding:20px 22px; color:#fff;
          animation: pop .5s both; box-shadow: 0 10px 30px rgba(0,0,0,.2);}
.verdict .vicon {font-size: clamp(2.2rem, 9vw, 3.2rem); line-height:1;}
.verdict h2 {color:#fff; margin:0; padding:0; font-size: clamp(1.15rem, 4.6vw, 1.7rem); font-weight:800;}
.verdict p {color:rgba(255,255,255,.92); margin:4px 0 0; font-size: clamp(.85rem, 3vw, 1rem);}
.v-healthy {background: linear-gradient(120deg,#1b5e20,#43a047,#9ccc65);}
.v-moderate {background: linear-gradient(120deg,#e65100,#ffa000);}
.v-high {background: linear-gradient(120deg,#b71c1c,#ef5350);}
.v-incurable {background: linear-gradient(120deg,#3e0a0a,#8e1b1b);}
.v-none {background: linear-gradient(120deg,#1b5e20,#43a047,#9ccc65);}

/* ---- confidence bar ---- */
.cbar {margin: 14px 0 6px; padding: 14px 16px; border-radius:16px; animation: pop .5s both;
       background: rgba(128,128,128,.08); border:1px solid rgba(128,128,128,.2);}
.cbar-top {display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px;}
.cbar-top span {opacity:.75; font-size:.9rem;} .cbar-top b {font-size:1.35rem;}
.cbar-track {height:16px; border-radius:99px; background: rgba(128,128,128,.25); overflow:hidden;}
.cbar-fill {height:100%; border-radius:99px; animation: grow 1.1s ease-out;}
.cbar-scale {display:flex; justify-content:space-between; font-size:.72rem; opacity:.55; margin-top:6px;}
.cbar-note {font-size:.85rem; margin-top:6px; opacity:.85;}

.plan-title {font-size: clamp(1.1rem, 4vw, 1.4rem); font-weight:800; margin: 22px 0 4px;}

/* ---- tabs / buttons ---- */
.stTabs [data-baseweb="tab-list"] {gap: 6px; flex-wrap: wrap;}
.stTabs [data-baseweb="tab"] {border-radius: 12px; padding: 8px 14px;}
.stButton > button, .stFormSubmitButton > button {border-radius: 14px; font-weight: 700; width: 100%;
       border: 0; color:#fff; background: linear-gradient(120deg,#1b8a3a,#7cb342); transition: transform .15s;}
.stButton > button:hover, .stFormSubmitButton > button:hover {transform: scale(1.02); color:#fff;}

/* ---- login ---- */
.lock {text-align:center; margin: 6vh auto 10px; max-width: 420px;}
.lock .icon {font-size: 3.4rem; animation: floaty 5s ease-in-out infinite; display:inline-block;}
.lock h2 {margin: 4px 0 2px; font-weight: 800;} .lock p {opacity:.7; margin:0;}

.foot {text-align:center; opacity:.6; font-size:.8rem; margin-top:28px;}

/* ---- phones ---- */
@media (max-width: 640px) {
  .block-container {padding: .6rem .6rem 2.5rem;}
  .hero {padding: 26px 14px 22px; border-radius: 20px;}
  .steps {gap:6px;} .step {font-size:.72rem; padding:10px 4px;} .step b {font-size:1.25rem;}
  .crops {gap:10px;}
  .verdict {padding: 16px 14px; gap: 12px; margin-top: 10px;}
  .stTabs [data-baseweb="tab"] {padding: 6px 10px; font-size:.85rem;}
}
</style>
"""

HIDE_SIDEBAR = "<style>[data-testid='stSidebar'], [data-testid='stSidebarCollapsedControl'], " \
               "[data-testid='stExpandSidebarButton'] {display:none !important;}</style>"

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

st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- password gate
def get_password() -> str:
    """Prefer Streamlit secrets (APP_PASSWORD); fall back to the default below."""
    try:
        return str(st.secrets["APP_PASSWORD"])
    except Exception:
        return "Rajesh444"


def check_password() -> bool:
    if st.session_state.get("auth_ok"):
        return True

    st.markdown(HIDE_SIDEBAR, unsafe_allow_html=True)
    st.markdown('<div class="lock"><div class="icon">🔒</div><h2>Private Access</h2>'
                '<p>Enter the password to open Plant Disease Detection</p></div>',
                unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        with st.form("login", clear_on_submit=True):
            pw = st.text_input("Password", type="password", placeholder="Enter password",
                               label_visibility="collapsed")
            go = st.form_submit_button("🔓 Unlock")
        if go:
            if hmac.compare_digest(pw.encode(), get_password().encode()):
                st.session_state["auth_ok"] = True
                st.rerun()
            else:
                time.sleep(1)  # slows brute-force guessing
                st.error("Wrong password. Please try again.")
    return False


if not check_password():
    st.stop()


# --------------------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading AI model (downloading if needed)...")
def load_model():
    import tensorflow as tf

    if not MODEL_PATH.exists():
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


# --------------------------------------------------------------------------- helpers
def pretty(class_name: str) -> str:
    crop, _, cond = class_name.partition("___")
    crop = crop.replace("_", " ").replace(",", "")
    cond = cond.replace("_", " ").strip()
    return f"{crop} - {cond}"


def crop_of(class_name: str) -> str:
    return class_name.split("___")[0].split("_")[0].split(",")[0]


def ask_ollama(question: str, context: str, model_name: str, host: str) -> str:
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


def crop_visual(name: str, svg: str) -> str:
    for ext in ("jpg", "jpeg", "png", "webp"):
        p = BASE_DIR / "images" / f"{name}.{ext}"
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'<img src="data:image/{ext};base64,{b64}">'
    return svg


def hero_and_crops():
    st.markdown(
        '<div class="hero"><span class="leaf l1">🍃</span><span class="leaf l2">🌿</span>'
        '<span class="leaf l3">🍀</span><h1>🌿 Plant Disease Detection</h1>'
        '<p>Snap or upload a leaf photo and get an instant AI diagnosis with treatment advice</p></div>',
        unsafe_allow_html=True)
    st.markdown(
        '<div class="steps"><div class="step"><b>📸</b>1. Add leaf photo</div>'
        '<div class="step"><b>🧠</b>2. AI analyses it</div>'
        '<div class="step"><b>💊</b>3. Get treatment plan</div></div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="crops">'
        f'<div class="crop">{crop_visual("tomato", TOMATO_SVG)}<b>Tomato</b><span>9 diseases + healthy</span></div>'
        f'<div class="crop">{crop_visual("potato", POTATO_SVG)}<b>Potato</b><span>2 diseases + healthy</span></div>'
        f'</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- sidebar
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
    st.divider()
    if st.button("🚪 Log out"):
        st.session_state["auth_ok"] = False
        st.rerun()

mode = PREPROCESS_MODES[mode_label]

# --------------------------------------------------------------------------- main page
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
    with st.spinner("🔬 Analysing leaf..."):
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

# --------------------------------------------------------------------------- verdict
supported = crop_of(best_name) in SUPPORTED_CROPS and info is not None
confident = conf >= threshold
severity = info["severity"] if supported else "none"
is_healthy = supported and severity == "none"

if conf >= 0.85:
    bar_color, conf_word = "linear-gradient(90deg,#2e7d32,#66bb6a)", "High confidence"
elif conf >= 0.65:
    bar_color, conf_word = "linear-gradient(90deg,#ef6c00,#ffb300)", "Medium confidence"
else:
    bar_color, conf_word = "linear-gradient(90deg,#c62828,#ef5350)", "Low confidence"

col_img, col_res = st.columns([1, 1.3], gap="large")
with col_img:
    st.image(image, caption="Your leaf", use_container_width=True)

with col_res:
    if not supported:
        st.markdown('<div class="verdict v-moderate"><div class="vicon">🤔</div><div>'
                    '<h2>Not a tomato or potato leaf</h2>'
                    f'<p>Looks like: {pretty(best_name)}</p></div></div>', unsafe_allow_html=True)
    elif not confident:
        st.markdown('<div class="verdict v-moderate"><div class="vicon">❓</div><div>'
                    '<h2>Result not clear</h2>'
                    '<p>The model is not sure about this photo. Please retake it.</p></div></div>',
                    unsafe_allow_html=True)
    elif is_healthy:
        st.markdown('<div class="verdict v-healthy"><div class="vicon">🌱</div><div>'
                    '<h2>Your plant is healthy</h2>'
                    '<p>No disease or issue was found on this leaf.</p></div></div>',
                    unsafe_allow_html=True)
    else:
        icon = {"moderate": "⚠️", "high": "🚨", "incurable": "🛑"}.get(severity, "⚠️")
        st.markdown(f'<div class="verdict v-{severity}"><div class="vicon">{icon}</div><div>'
                    '<h2>Your plant has a disease</h2>'
                    f'<p>Detected: <b>{info["title"]}</b></p></div></div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="cbar"><div class="cbar-top"><span>Model confidence</span><b>{conf:.1%}</b></div>'
        f'<div class="cbar-track"><div class="cbar-fill" style="width:{conf*100:.1f}%; background:{bar_color}"></div></div>'
        f'<div class="cbar-scale"><span>0%</span><span>50%</span><span>100%</span></div>'
        f'<div class="cbar-note">{conf_word}</div></div>', unsafe_allow_html=True)

with st.expander("📊 Top 3 possibilities"):
    top = pd.DataFrame({"Class": [pretty(CLASS_NAMES[i]) for i in order[:3]],
                        "Probability": [probs[i] for i in order[:3]]}).set_index("Class")
    st.bar_chart(top, horizontal=True)

if not supported:
    st.warning("This app only covers tomato and potato leaves. Please upload one of those.")
    st.stop()
if not confident:
    st.warning("The model is **not confident** about this photo. Retake it closer, in daylight, "
               "with one leaf filling most of the frame, and avoid blur or heavy shadows.")
    st.stop()

# --------------------------------------------------------------------------- about + treatment plan (below)
st.markdown(f'<div class="card"><b>What it is</b><br>{info["cause"]}</div>', unsafe_allow_html=True)

plan_title = "🌱 Care plan to keep it healthy" if is_healthy else "💊 Treatment plan"
st.markdown(f'<div class="plan-title">{plan_title}</div>', unsafe_allow_html=True)

t_treat, t_sym, t_prev = st.tabs(["💊 Treatment", "🔍 Symptoms", "🛡️ Prevention"])
with t_treat:
    st.markdown("🌿 **Home / Organic Remedies**")
    bullets(info["cultural"])
    if info["chemical"]:
        st.markdown("💊 **Commercial Treatment & Dosages**")
        bullets(info["chemical"])
        st.markdown("🛒 **Local Store / Online Search Tip:** Search the medicine name above on agricultural "
                    "supply stores or delivery apps (like Coupang or local garden supply centers) to find "
                    "available brands easily.")
        st.caption("Always follow the product label for exact safety precautions, mixing ratios, "
                   "and local agricultural regulations.")
with t_sym:
    bullets(info["symptoms"])
with t_prev:
    bullets(info["prevention"])

# --------------------------------------------------------------------------- optional AI advisor
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
