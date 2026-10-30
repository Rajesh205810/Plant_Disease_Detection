"""Plant Disease Detection & Open-Source AI Assistant - Streamlit App.

Run:  streamlit run app.py
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

# Session State Setup
if "view" not in st.session_state:
    st.session_state.view = "scanner"
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "current_diagnosis" not in st.session_state:
    st.session_state.current_diagnosis = None
if "show_welcome_popup" not in st.session_state:
    st.session_state.show_welcome_popup = True

# CSS Styling (Includes Hover Highlighting and Hides Streamlit Cloud Chrome)
CSS = """
<style>
#MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stToolbar"],
[data-testid="stDeployButton"], [data-testid="stAppDeployButton"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {display: none !important; visibility: hidden !important;}
header[data-testid="stHeader"] {background: transparent !important;}

.block-container {padding: 1.2rem 1rem 3rem; max-width: 880px;}
@keyframes flow {0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%}}
@keyframes floaty {0%,100%{transform:translateY(0) rotate(-6deg)} 50%{transform:translateY(-10px) rotate(6deg)}}
@keyframes pop {from{opacity:0; transform:translateY(14px)} to{opacity:1; transform:none}}

.hero {position: relative; overflow: hidden; border-radius: 26px; padding: 34px 22px 28px;
       text-align: center; color: #fff; margin-bottom: 18px;
       background: linear-gradient(120deg,#0b3d1e,#1b8a3a,#7cb342,#1b8a3a,#0b3d1e);
       background-size: 300% 300%; animation: flow 14s ease infinite;
       box-shadow: 0 14px 40px rgba(27,138,58,.35);}
.hero h1 {color:#fff; margin:0; font-size: clamp(1.6rem, 5.5vw, 2.5rem); font-weight: 800;}
.hero p {color:#eaf7ea; margin:8px auto 0; max-width: 560px; font-size: clamp(.9rem, 2.8vw, 1.05rem);}
.hero .leaf {position:absolute; font-size: 2.4rem; opacity:.22; animation: floaty 6s ease-in-out infinite;}
.hero .l1 {left: 5%; top: 14%;} .hero .l2 {right: 6%; top: 40%; animation-delay: 1.5s; font-size:3rem;}
.hero .l3 {left: 20%; bottom: 6%; animation-delay: 3s; font-size:1.8rem;}

.welcome-popup {
    background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    border: 2px solid #4caf50; border-radius: 20px; padding: 20px 24px;
    margin-bottom: 20px; box-shadow: 0 6px 20px rgba(76,175,80,0.15); animation: pop 0.4s ease-out;
}
.welcome-popup h3 {color: #1b5e20; margin: 0 0 6px 0; font-size: 1.35rem;}
.welcome-popup p {color: #2e7d32; margin: 0; font-size: 0.98rem; line-height: 1.4;}

.steps {display:grid; grid-template-columns: repeat(3,1fr); gap:10px; margin-bottom:18px;}
.step {border-radius:16px; padding:12px 8px; text-align:center; font-size:.85rem;
       background: rgba(76,175,80,.10); border:1px solid rgba(76,175,80,.30); animation: pop .6s both;}
.step b {display:block; font-size:1.5rem;}

/* Crop Cards with Hover Highlighting Animation */
.crops {display:grid; grid-template-columns: 1fr 1fr; gap:14px; margin-bottom:20px;}
.crop {border-radius:20px; padding:16px 10px 14px; text-align:center; 
       transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease, background 0.3s ease;
       background: linear-gradient(160deg, rgba(76,175,80,.16), rgba(76,175,80,.04));
       border:1px solid rgba(76,175,80,.35); backdrop-filter: blur(6px); cursor: pointer;}
.crop:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 12px 30px rgba(76,175,80,.35);
    border-color: #4caf50;
    background: linear-gradient(160deg, rgba(76,175,80,.26), rgba(76,175,80,.1));
}
.crop svg, .crop img {width: min(96px, 24vw); height: min(96px, 24vw); object-fit: cover; border-radius: 16px; transition: transform 0.3s ease;}
.crop:hover svg, .crop:hover img {transform: scale(1.08);}
.crop b {display:block; margin-top:8px; font-size:1.05rem;}
.crop span {font-size:.82rem; opacity:.75;}

.card {border-radius:16px; padding:14px 18px; margin:8px 0; background: rgba(128,128,128,.08); border:1px solid rgba(128,128,128,.2);}
.verdict {display:flex; align-items:center; gap:16px; border-radius:22px; padding:20px 22px; color:#fff; box-shadow: 0 10px 30px rgba(0,0,0,.2);}
.verdict .vicon {font-size: clamp(2.2rem, 9vw, 3.2rem); line-height:1;}
.verdict h2 {color:#fff; margin:0; font-size: clamp(1.15rem, 4.6vw, 1.7rem); font-weight:800;}
.v-healthy {background: linear-gradient(120deg,#1b5e20,#43a047,#9ccc65);}
.v-moderate {background: linear-gradient(120deg,#e65100,#ffa000);}
.v-high {background: linear-gradient(120deg,#b71c1c,#ef5350);}
.v-incurable {background: linear-gradient(120deg,#3e0a0a,#8e1b1b);}

.cbar {margin: 14px 0 6px; padding: 14px 16px; border-radius:16px; background: rgba(128,128,128,.08); border:1px solid rgba(128,128,128,.2);}
.cbar-top {display:flex; justify-content:space-between; align-items:baseline; margin-bottom:8px;}
.cbar-track {height:16px; border-radius:99px; background: rgba(128,128,128,.25); overflow:hidden;}
.cbar-fill {height:100%; border-radius:99px;}
.cbar-scale {display:flex; justify-content:space-between; font-size:.72rem; opacity:.55; margin-top:6px;}

.ai-context-banner {
    background: linear-gradient(135deg, rgba(27,138,58,0.18), rgba(124,179,66,0.1));
    border: 1px solid rgba(76,175,80,0.4);
    border-radius: 18px; padding: 14px 18px; margin: 10px 0 16px 0;
}
.plan-title {font-size: clamp(1.1rem, 4vw, 1.4rem); font-weight:800; margin: 22px 0 4px;}
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

# Password Protection Gate
def get_password() -> str:
    try:
        return str(st.secrets["APP_PASSWORD"])
    except Exception:
        return "Rajesh444"

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
            if hmac.compare_digest(pw.encode(), get_password().encode()):
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

def fetch_open_source_data(query: str, disease_name: str = "") -> str:
    search_term = f"{disease_name} {query} agriculture plant care".strip()
    fetched_info = ""
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(search_term)}&format=json&no_html=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=6) as response:
            data = json.loads(response.read().decode("utf-8"))
            if data.get("AbstractText"):
                fetched_info = data["AbstractText"]
            elif data.get("RelatedTopics"):
                snippets = [item["Text"] for item in data["RelatedTopics"] if isinstance(item, dict) and "Text" in item]
                if snippets: fetched_info = " ".join(snippets[:2])
    except Exception:
        pass
    return fetched_info

def generate_open_source_ai_response(user_query: str, diagnosis_ctx: dict, hf_token: str = "") -> str:
    dis_title = diagnosis_ctx.get("disease", "") if diagnosis_ctx else ""
    dis_cause = diagnosis_ctx.get("cause", "") if diagnosis_ctx else ""
    open_knowledge = fetch_open_source_data(user_query, dis_title)

    if hf_token.strip():
        try:
            hf_url = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-72B-Instruct"
            prompt = f"System: You are an agricultural expert.\nContext: {dis_title} - {dis_cause}\nOpen Knowledge: {open_knowledge}\nUser: {user_query}\nAnswer:"
            payload = json.dumps({"inputs": prompt, "parameters": {"max_new_tokens": 250}}).encode("utf-8")
            req = urllib.request.Request(hf_url, data=payload, headers={"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
                    return f"🌐 **Open-Source AI Response (Hugging Face):**\n\n{result[0]['generated_text']}"
        except Exception:
            pass

    q_lower = user_query.lower()
    if "eat" in q_lower or "harvest" in q_lower or "safe" in q_lower:
        if diagnosis_ctx and diagnosis_ctx.get('is_healthy'):
            return "✅ **Yes!** Your plant is completely healthy, so harvested produce is safe to eat."
        return f"⚠️ **Harvest Safety Advice:** Remove heavily decayed sections. Healthy produce can be consumed after washing.\n\n*Open Knowledge:* {open_knowledge if open_knowledge else 'Ensure harvested fruit is clean.'}"
    if "organic" in q_lower or "home" in q_lower or "remedy" in q_lower:
        if diagnosis_ctx and diagnosis_ctx.get('cultural'):
            remedies = "\n- ".join(diagnosis_ctx['cultural'])
            return f"🌿 **Organic Care Plan for {dis_title}:**\n\n- {remedies}\n\n🌐 *Open Data:* {open_knowledge}"
        return "🌿 Prune crowded leaves and avoid overhead watering."
    if diagnosis_ctx:
        return f"🤖 **AI Analysis for {dis_title}:**\n\n- **Cause:** {dis_cause}\n🌐 **Open Data:** {open_knowledge}"
    return open_knowledge if open_knowledge else "🤖 Upload a leaf image in the **Scanner** screen first!"

def bullets(items):
    for item in items: st.markdown(f"- {item}")

def crop_visual(name: str, svg: str) -> str:
    for ext in ("jpg", "jpeg", "png", "webp"):
        p = BASE_DIR / "images" / f"{name}.{ext}"
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'<img src="data:image/{ext};base64,{b64}">'
    return svg

def hero_and_crops():
    st.markdown('<div class="hero"><h1>🌿 Plant Disease Detection</h1><p>Snap or upload a leaf photo and get an instant AI diagnosis</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="steps"><div class="step"><b>📸</b>1. Add photo</div><div class="step"><b>🧠</b>2. AI analyses</div><div class="step"><b>💊</b>3. Treatment plan</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="crops"><div class="crop">{crop_visual("tomato", TOMATO_SVG)}<b>Tomato</b><span>10 diseases + healthy</span></div><div class="crop">{crop_visual("potato", POTATO_SVG)}<b>Potato</b><span>2 diseases + healthy</span></div></div>', unsafe_allow_html=True)

# Sidebar & Navigation
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
    st.header("⚙️ Settings & Confidence")
    threshold = st.slider("Minimum confidence score", 40, 95, 70, 5, format="%d%%") / 100
    
    with st.expander("Advanced & Open-Source AI"):
        mode_label = st.selectbox("Image scaling", list(PREPROCESS_MODES))
        hf_token = st.text_input("HuggingFace Open API Key", type="password", help="Optional token for advanced open-source LLM answers.")

    st.divider()
    if st.button("🚪 Log out"):
        st.session_state["auth_ok"] = False
        st.rerun()

mode = PREPROCESS_MODES[mode_label]

# SCREEN 1: LEAF SCANNER
if st.session_state.view == "scanner":
    if st.session_state.show_welcome_popup:
        col_p1, col_p2 = st.columns([12, 1])
        with col_p1:
            st.markdown(
                """
                <div class="welcome-popup">
                    <h3>🌱 Welcome to Our Project!</h3>
                    <p>Detect plant diseases in tomatoes and potatoes instantly. Upload an image below to get started.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col_p2:
            if st.button("❌", help="Close welcome popup"):
                st.session_state.show_welcome_popup = False
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
            "disease": info["title"], "confidence": conf, "cause": info["cause"],
            "symptoms": info["symptoms"], "cultural": info["cultural"],
            "chemical": info["chemical"], "prevention": info["prevention"], "is_healthy": is_healthy
        }

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
            st.markdown('<div class="verdict v-moderate"><div class="vicon">🤔</div><div><h2>Not a tomato or potato leaf</h2></div></div>', unsafe_allow_html=True)
        elif not confident:
            st.markdown('<div class="verdict v-moderate"><div class="vicon">❓</div><div><h2>Result not clear</h2></div></div>', unsafe_allow_html=True)
        elif is_healthy:
            st.markdown('<div class="verdict v-healthy"><div class="vicon">🌱</div><div><h2>Your plant is healthy</h2></div></div>', unsafe_allow_html=True)
        else:
            icon = {"moderate": "⚠️", "high": "🚨", "incurable": "🛑"}.get(severity, "⚠️")
            st.markdown(f'<div class="verdict v-{severity}"><div class="vicon">{icon}</div><div><h2>Disease Detected</h2><p><b>{info["title"]}</b></p></div></div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="cbar"><div class="cbar-top"><span>Model confidence</span><b>{conf:.1%}</b></div>'
            f'<div class="cbar-track"><div class="cbar-fill" style="width:{conf*100:.1f}%; background:{bar_color}"></div></div>'
            f'<div class="cbar-scale"><span>0%</span><span>50%</span><span>100%</span></div>'
            f'</div>', unsafe_allow_html=True)

        if supported and confident:
            if st.button("💬 Ask AI About This Diagnosis", use_container_width=True):
                st.session_state.view = "ai_assistant"
                st.rerun()

    with st.expander("📊 Top 3 possibilities"):
        top = pd.DataFrame({"Class": [pretty(CLASS_NAMES[i]) for i in order[:3]], "Probability": [probs[i] for i in order[:3]]}).set_index("Class")
        st.bar_chart(top, horizontal=True)

    if not supported or not confident:
        st.stop()

    st.markdown(f'<div class="card"><b>What it is</b><br>{info["cause"]}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="plan-title">Treatment plan</div>', unsafe_allow_html=True)

    t_treat, t_sym, t_prev = st.tabs(["💊 Treatment", "🔍 Symptoms", "🛡️ Prevention"])
    with t_treat:
        st.markdown("🌿 **Home / Organic Remedies**")
        bullets(info["cultural"])
        if info["chemical"]:
            st.markdown("💊 **Commercial Treatment & Dosages**")
            bullets(info["chemical"])
    with t_sym:
        bullets(info["symptoms"])
    with t_prev:
        bullets(info["prevention"])

    st.markdown('<div class="foot">Decision aid only - confirm serious outbreaks with local experts.</div>', unsafe_allow_html=True)

# SCREEN 2: IN-APP OPEN-SOURCE AI ASSISTANT
elif st.session_state.view == "ai_assistant":
    st.markdown("## 🤖 In-App Open-Source AI Assistant")
    st.caption("Answers are dynamically fetched from open sources & synthesized in the app backend.")

    curr_diag = st.session_state.current_diagnosis
    if curr_diag:
        st.markdown(
            f'<div class="ai-context-banner">'
            f'<b>📌 Active Scan Context:</b> {curr_diag["disease"]} '
            f'({curr_diag["confidence"]:.1%} confidence)<br>'
            f'<small>Scan details are automatically referenced for your questions.</small>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("💡 No leaf scan loaded yet. You can ask general plant questions or upload a photo in the Scanner.")

    st.write("**Quick Questions:**")
    qcol1, qcol2, qcol3 = st.columns(3)
    quick_prompt = None
    with qcol1:
        if st.button("🌿 Organic Remedies"): quick_prompt = "What organic remedies should I apply?"
    with qcol2:
        if st.button("⚠️ Safe to Harvest?"): quick_prompt = "Is produce from this plant safe to eat?"
    with qcol3:
        if st.button("💊 Spray Advice"): quick_prompt = "What spraying precautions should I follow?"

    st.divider()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask any question about plant health or treatments...")
    active_query = user_input or quick_prompt

    if active_query:
        st.session_state.chat_messages.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        with st.chat_message("assistant"):
            with st.spinner("Fetching open-source reference data & synthesizing response..."):
                token_val = hf_token if 'hf_token' in locals() else ""
                response = generate_open_source_ai_response(active_query, curr_diag, token_val)
                st.markdown(response)

        st.session_state.chat_messages.append({"role": "assistant", "content": response})

    st.divider()
    if st.button("⬅️ Back to Leaf Scanner"):
        st.session_state.view = "scanner"
        st.rerun()
