"""Plant Disease Detection & Open-Source AI Assistant - Streamlit App.

Run:  streamlit run app.py
Needs: disease_info.py next to this file, and model.h5 (auto-downloads if missing).
"""
import base64
import hmac
import io
import json
import time
import urllib.parse
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

st.set_page_config(
    page_title="Plant Disease Detection & Open-Source AI",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------- Session State Setup
if "view" not in st.session_state:
    st.session_state.view = "scanner"
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "current_diagnosis" not in st.session_state:
    st.session_state.current_diagnosis = None
if "show_welcome" not in st.session_state:
    st.session_state.show_welcome = True

# --------------------------------------------------------------------------- CSS Styling
CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding: 1.2rem 1rem 3rem; max-width: 880px;}
@keyframes flow {0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%}}
@keyframes floaty {0%,100%{transform:translateY(0) rotate(-6deg)} 50%{transform:translateY(-10px) rotate(6deg)}}
@keyframes pop {from{opacity:0; transform:translateY(14px)} to{opacity:1; transform:none}}

/* ---- hero banner ---- */
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

/* ---- enlarged welcome banner ---- */
.welcome-banner {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    border: 2px solid #4caf50;
    border-radius: 22px;
    padding: 24px 26px;
    margin-bottom: 22px;
    box-shadow: 0 8px 24px rgba(76,175,80,0.18);
    animation: pop 0.5s ease-out;
}
.welcome-banner h2 {color: #1b5e20; margin: 0 0 8px 0; font-size: 1.6rem; font-weight: 800;}
.welcome-banner p {color: #2e7d32; margin: 0; font-size: 1.05rem; line-height: 1.5;}

/* ---- steps ---- */
.steps {display:grid; grid-template-columns: repeat(3,1fr); gap:10px; margin-bottom:18px;}
.step {border-radius:16px; padding:12px 8px; text-align:center; font-size:.85rem;
       background: rgba(76,175,80,.10); border:1px solid rgba(76,175,80,.30); animation: pop .6s both;}
.step b {display:block; font-size:1.5rem;}

/* ---- crop cards ---- */
.crops {display:grid; grid-template-columns: 1fr 1fr; gap:14px; margin-bottom:20px;}
.crop {border-radius:20px; padding:16px 10px 14px; text-align:center;
       background: linear-gradient(160deg, rgba(76,175,80,.16), rgba(76,175,80,.04));
       border:1px solid rgba(76,175,80,.35); backdrop-filter: blur(6px);}
.crop svg, .crop img {width: min(96px, 24vw); height: min(96px, 24vw); object-fit: cover; border-radius: 16px;}
.crop b {display:block; margin-top:8px; font-size:1.05rem; color: #1b5e20;}
.crop span {font-size:.82rem; opacity:.75;}

/* ---- result cards & verdict ---- */
.card {border-radius:16px; padding:14px 18px; margin:8px 0; background: rgba(128,128,128,.08); border:1px solid rgba(128,128,128,.2);}
.verdict {display:flex; align-items:center; gap:16px; border-radius:22px; padding:20px 22px; color:#fff; box-shadow: 0 10px 30px rgba(0,0,0,.2);}
.verdict .vicon {font-size: clamp(2.2rem, 9vw, 3.2rem); line-height:1;}
.verdict h2 {color:#fff; margin:0; font-size: clamp(1.15rem, 4.6vw, 1.7rem); font-weight:800;}
.v-healthy {background: linear-gradient(120deg,#1b5e20,#43a047,#9ccc65);}
.v-moderate {background: linear-gradient(120deg,#e65100,#ffa000);}
.v-high {background: linear-gradient(120deg,#b71c1c,#ef5350);}
.v-incurable {background: linear-gradient(120deg,#3e0a0a,#8e1b1b);}

.solution-box {background: #f1f8e9; border: 2px solid #aed581; border-radius: 16px; padding: 20px; margin: 16px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.03);}
.solution-box h3 {color: #2e7d32; margin-top: 0; font-size: 1.25rem;}
.stat {text-align: center; border-radius: 16px; padding: 14px; background: #ffffff; border: 2px solid #c8e6c9;}
.stat small {color: #558b2f; display: block; font-weight: 600;} .stat b {font-size: 1.3rem; color: #1b5e20;}

.lock {text-align:center; margin: 6vh auto 10px; max-width: 420px;}
.lock .icon {font-size: 3.4rem; display:inline-block;}
.lock h2 {margin: 4px 0 2px; font-weight: 800;} .lock p {opacity:.7; margin:0;}
.foot {text-align:center; opacity:.6; font-size:.8rem; margin-top:28px;}
</style>
"""

HIDE_SIDEBAR = "<style>[data-testid='stSidebar'] {display:none !important;}</style>"
TOMATO_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="56" r="34" fill="#e53935"/><path d="M50 24 L43 14 L50 19 L57 14 Z" fill="#2e7d32"/><rect x="48" y="10" width="4" height="12" rx="2" fill="#1b5e20"/></svg>"""
POTATO_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><path d="M18 58 C14 36 34 22 56 26 C78 30 90 46 82 66 C74 84 40 88 26 76 Z" fill="#c8923f"/></svg>"""

st.markdown(CSS, unsafe_allow_html=True)

# Password Gate
def check_password() -> bool:
    if st.session_state.get("auth_ok"):
        return True
    st.markdown(HIDE_SIDEBAR, unsafe_allow_html=True)
    st.markdown('<div class="lock"><div class="icon">🔒</div><h2>Private Access</h2><p>Enter password to open Plant Disease Detection</p></div>', unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        with st.form("login", clear_on_submit=True):
            pw = st.text_input("Password", type="password", placeholder="Enter password", label_visibility="collapsed")
            go = st.form_submit_button("🔓 Unlock")
        if go:
            if hmac.compare_digest(pw.encode(), "Rajesh444".encode()):
                st.session_state["auth_ok"] = True
                st.rerun()
            else:
                time.sleep(1)
                st.error("Wrong password. Please try again.")
    return False

if not check_password():
    st.stop()

# Model and Helpers
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    import tensorflow as tf
    if not MODEL_PATH.exists():
        gdrive_url = "https://drive.google.com/uc?id=1wuHFBZs-76upHrPybchq4LZbh8khOTAP&export=download"
        gdown.download(gdrive_url, str(MODEL_PATH), quiet=False)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess(img: Image.Image, mode: str) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
    arr = np.asarray(img, dtype="float32")
    if mode == "div255": arr = arr / 255.0
    elif mode == "xception": arr = arr / 127.5 - 1.0
    return np.expand_dims(arr, 0)

@st.cache_data(show_spinner=False, max_entries=32)
def predict(image_bytes: bytes, mode: str) -> list:
    model = load_model()
    img = Image.open(io.BytesIO(image_bytes))
    probs = model.predict(preprocess(img, mode), verbose=0)[0]
    return [float(p) for p in probs]

def pretty(class_name: str) -> str:
    crop, _, cond = class_name.partition("___")
    return f"{crop.replace('_', ' ')} - {cond.replace('_', ' ').strip()}"

def crop_of(class_name: str) -> str:
    return class_name.split("___")[0].split("_")[0]

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
    # --------------------------------------------------------------------------- Sidebar & Navigation
with st.sidebar:
    st.header("🧭 App Navigation")
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        if st.button("🌿 Scanner", use_container_width=True):
            st.session_state.view = "scanner"
            st.rerun()
    with col_v2:
        if st.button("🤖 AI Chat", use_container_width=True):
            st.session_state.view = "ai_assistant"
            st.rerun()

    st.divider()
    st.header("⚙️ Settings")
    threshold = st.slider("Minimum confidence", 40, 95, 70, 5, format="%d%%") / 100
    with st.expander("Advanced"):
        mode_label = st.selectbox("Image scaling used in training", list(PREPROCESS_MODES))

    st.divider()
    if st.button("🚪 Log out"):
        st.session_state["auth_ok"] = False
        st.rerun()

mode = PREPROCESS_MODES[mode_label]

# =========================================================================== SCREEN 1: SCANNER
if st.session_state.view == "scanner":
    # Enlarged Welcome Slide with a Clear Cross (Dismiss) Button
    if st.session_state.show_welcome:
        col_w1, col_w2 = st.columns([10, 1])
        with col_w1:
            st.markdown(
                """
                <div class="welcome-banner">
                    <h2>🌱 Welcome to Plant Disease Detection Project!</h2>
                    <p>
                        This intelligent diagnostic platform helps home gardeners and farmers instantly identify crop ailments in <b>tomatoes and potatoes</b>. 
                        Get precise AI confidence scores, store-bought chemical treatments, and organic home remedies instantly.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_w2:
            if st.button("❌", help="Clear welcome slide"):
                st.session_state.show_welcome = False
                st.rerun()

    hero_and_crops()

    tab_up, tab_cam = st.tabs(["📁 Upload photo", "📷 Use camera"])
    with tab_up:
        uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    with tab_cam:
        camera = st.camera_input("Take a photo of the leaf", label_visibility="collapsed")

    source = uploaded or camera
    if source is None:
        st.info("💡 Tip: use a close, well-lit photo of ONE leaf on a plain background.")
        st.markdown('<div class="foot">Decision aid only - confirm serious outbreaks with local experts.</div>', unsafe_allow_html=True)
        st.stop()

    image_bytes = source.getvalue()
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    probs = predict(image_bytes, mode)

    order = np.argsort(probs)[::-1]
    best = int(order[0])
    best_name, conf = CLASS_NAMES[best], probs[best]
    info = DISEASE_INFO.get(best_name)

    supported = crop_of(best_name) in SUPPORTED_CROPS and info is not None
    confident = conf >= threshold
    severity = info["severity"] if supported else "none"
    is_healthy = supported and severity == "none"

    if supported and confident:
        st.session_state.current_diagnosis = {
            "disease": info["title"],
            "confidence": conf,
            "cause": info["cause"],
            "symptoms": info["symptoms"],
            "cultural": info["cultural"],
            "chemical": info["chemical"],
            "prevention": info["prevention"],
            "is_healthy": is_healthy
        }

    col_img, col_res = st.columns([1, 1.3], gap="large")
    with col_img:
        st.image(image, caption="Your leaf", use_container_width=True)

    with col_res:
        if not supported:
            st.markdown('<div class="verdict v-moderate"><div class="vicon">🤔</div><div><h2>Not a tomato or potato leaf</h2></div></div>', unsafe_allow_html=True)
        elif not confident:
            st.markdown('<div class="verdict v-moderate"><div class="vicon">❓</div><div><h2>Result not clear</h2></div></div>', unsafe_allow_html=True)
        elif is_healthy:
            st.markdown('<div class="verdict v-healthy"><div class="vicon">🌱</div><div><h2>Your plant is healthy</h2></div></div>', unsafe_allow_html=True)
        else:
            icon = {"moderate": "⚠️", "high": "🚨", "incurable": "🛑"}.get(severity, "⚠️")
            st.markdown(f'<div class="verdict v-{severity}"><div class="vicon">{icon}</div><div><h2>Disease Detected</h2><p><b>{info["title"]}</b></p></div></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="stat"><small>Confidence</small><b>{conf:.1%}</b></div>', unsafe_allow_html=True)
        
        if supported and confident:
            if st.button("💬 Ask AI About This Diagnosis", use_container_width=True):
                st.session_state.view = "ai_assistant"
                st.rerun()

    if not supported or not confident:
        st.stop()

    # Solution Box Directly Under Results
    st.markdown(f"""
    <div class="solution-box">
        <h3>📋 Action Plan & Prescriptions</h3>
        <p><b>What it is:</b> {info['cause']}</p>
        <hr style="border: 0; border-top: 1px solid #c8e6c9; margin: 10px 0;">
        <p><b>💊 Store-Bought Chemical Treatment:</b><br>{' • '.join(info['chemical']) if info['chemical'] else 'Consult local store'}</p>
        <p><b>🌿 Homemade / Organic Remedy:</b><br>{' • '.join(info['cultural'])}</p>
    </div>
    """, unsafe_allow_html=True)

# =========================================================================== SCREEN 2: AI ASSISTANT
elif st.session_state.view == "ai_assistant":
    st.markdown("## 🤖 In-App Plant AI Assistant")
    st.write("Ask any questions regarding your plant's condition or treatment options.")
    
    user_query = st.text_input("Ask a question...")
    if user_query:
        st.success(f"AI Assistant Response: Follow product safety labels for chemical applications or use organic neem oil weekly.")
    
    if st.button("⬅️ Back to Scanner"):
        st.session_state.view = "scanner"
        st.rerun()
