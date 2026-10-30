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
    page_title="Plant Disease Detection",
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
    st.session_state.show_welcome = True  # Controls the welcome banner visibility

# --------------------------------------------------------------------------- CSS Styling
CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding: 1.2rem 1rem 3rem; max-width: 880px;}
.hero {position: relative; overflow: hidden; border-radius: 26px; padding: 34px 22px 28px;
       text-align: center; color: #fff; margin-bottom: 18px;
       background: linear-gradient(120deg,#0b3d1e,#1b8a3a,#7cb342,#1b8a3a,#0b3d1e);
       background-size: 300% 300%; box-shadow: 0 14px 40px rgba(27,138,58,.35);}
.hero h1 {color:#fff; margin:0; font-size: clamp(1.6rem, 5.5vw, 2.5rem); font-weight: 800; letter-spacing:.3px;}
.hero p {color:#eaf7ea; margin:8px auto 0; max-width: 560px; font-size: clamp(.9rem, 2.8vw, 1.05rem);}
.welcome-banner {background: #e8f5e9; border: 2px solid #66bb6a; border-radius: 18px; padding: 16px 20px; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);}
.welcome-header {display: flex; justify-content: space-between; align-items: center;}
.crops {display:grid; grid-template-columns: 1fr 1fr; gap:14px; margin-bottom:20px;}
.crop {border-radius:20px; padding:16px 10px 14px; text-align:center;
       background: linear-gradient(160deg, rgba(76,175,80,.16), rgba(76,175,80,.04));
       border:1px solid rgba(76,175,80,.35);}
.crop svg, .crop img {width: min(96px, 24vw); height: min(96px, 24vw); object-fit: cover; border-radius: 16px;}
.crop b {display:block; margin-top:8px; font-size:1.05rem; color: #1b5e20;}
.crop span {font-size:.82rem; opacity:.75;}
.card {border-radius:16px; padding:14px 18px; margin:8px 0; background: rgba(128,128,128,.08); border:1px solid rgba(128,128,128,.2);}
.solution-box {background: #f1f8e9; border: 2px solid #aed581; border-radius: 16px; padding: 20px; margin: 16px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.03);}
.solution-box h3 {color: #2e7d32; margin-top: 0; font-size: 1.25rem;}
.banner {border-radius: 16px; padding: 16px 20px; margin: 14px 0 18px; color: #fff; font-weight: 700; font-size: 1.15rem;}
.b-none {background: #2e7d32;} .b-moderate {background: #f57c00;} .b-high {background: #e53935;} .b-incurable {background: #b71c1c;}
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

# Password Gate Check
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
                st.error("Wrong password. Try again.")
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
    # --------------------------------------------------------------------------- Sidebar Navigation & Settings
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
        mode_label = st.selectbox("Image scaling", list(PREPROCESS_MODES))

    st.divider()
    if st.button("🚪 Log out"):
        st.session_state["auth_ok"] = False
        st.rerun()

mode = PREPROCESS_MODES[mode_label]

# =========================================================================== MAIN APP SCREENS
if st.session_state.view == "scanner":
    # Welcome banner with close button (Cross)
    if st.session_state.show_welcome:
        st.markdown(
            """
            <div class="welcome-banner">
                <div class="welcome-header">
                    <h3 style="margin:0; color:#1b5e20;">🌱 Welcome to Our Project!</h3>
                </div>
                <p style="margin:6px 0 0 0; color:#2e7d32;">
                    This app detects tomato and potato plant diseases instantly, offering professional store-bought treatments and organic home remedies. Upload a leaf image below to begin!
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("✕ Dismiss Welcome Slide"):
            st.session_state.show_welcome = False
            st.rerun()

    st.markdown('<div class="hero"><h1>🌿 Plant Disease Detection</h1><p>Upload a leaf photo for instant AI diagnosis and care plans</p></div>', unsafe_allow_html=True)
    
    tab_up, tab_cam = st.tabs(["📁 Upload Photo", "📷 Use Camera"])
    with tab_up:
        uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    with tab_cam:
        camera = st.camera_input("Take a photo of the leaf", label_visibility="collapsed")

    source = uploaded or camera
    if source is None:
        st.info("💡 Tip: use a close, well-lit photo of ONE leaf on a plain background.")
        st.stop()

    image_bytes = source.getvalue()
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    probs = predict(image_bytes, mode)

    order = np.argsort(probs)[::-1]
    best = int(order[0])
    best_name, conf = CLASS_NAMES[best], probs[best]
    info = DISEASE_INFO.get(best_name, {})

    col_img, col_res = st.columns([1, 1.3], gap="large")
    with col_img:
        st.image(image, caption="Your Leaf Sample", use_container_width=True)
    with col_res:
        st.markdown(f'<div class="stat"><small>Prediction</small><b>{pretty(best_name)}</b></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="stat" style="margin-top:10px"><small>Confidence</small><b>{conf:.1%}</b></div>', unsafe_allow_html=True)
        st.progress(min(max(conf, 0.0), 1.0))

    if conf >= threshold:
        severity = info.get("severity", "moderate")
        label = {"none": "✅ Healthy leaf - no treatment needed",
                 "moderate": f"⚠️ {info.get('title', best_name)} - Treat promptly",
                 "high": f"🚨 {info.get('title', best_name)} - Spreads fast, act today",
                 "incurable": f"🛑 {info.get('title', best_name)} - Protect surrounding crops"}[severity]
        
        st.markdown(f'<div class="banner b-{severity}">{label}</div>', unsafe_allow_html=True)
        
        # Treatment Action Plan directly under results
        st.markdown(f"""
        <div class="solution-box">
            <h3>📋 Action Plan & Prescriptions</h3>
            <p><b>What it is:</b> {info.get('cause', 'N/A')}</p>
            <hr style="border: 0; border-top: 1px solid #c8e6c9; margin: 10px 0;">
            <p><b>💊 Store-Bought Chemical Treatment:</b><br>{' • '.join(info.get('chemical', ['Consult local agricultural store']))}</p>
            <p><b>🌿 Homemade / Organic Remedy:</b><br>{' • '.join(info.get('cultural', ['Maintain clean spacing and airflow']))}</p>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.view == "ai_assistant":
    st.markdown("## 🤖 In-App Plant AI Assistant")
    st.write("Ask any questions regarding your plant's condition or treatment options.")
    
    user_query = st.text_input("Ask a question...")
    if user_query:
        st.success(f"AI Assistant Response: To treat your plant effectively, ensure you follow product safety labels or apply organic neem oil regularly.")
    
    if st.button("⬅️ Back to Scanner"):
        st.session_state.view = "scanner"
        st.rerun()
