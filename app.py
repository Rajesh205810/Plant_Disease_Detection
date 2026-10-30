"""Plant Disease Detection & In-App AI Assistant - Streamlit App.

Run:  streamlit run app.py
Needs: disease_info.py next to this file, and model.h5 (auto-downloads if missing).
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

st.set_page_config(
    page_title="Plant Disease Detection & AI Assistant",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------------------------------- Session State Initialization
if "view" not in st.session_state:
    st.session_state.view = "scanner"  # "scanner" or "ai_assistant"
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "current_diagnosis" not in st.session_state:
    st.session_state.current_diagnosis = None

# --------------------------------------------------------------------------- Styling
CSS = """
<style>
/* ---- hide Streamlit toolbar and footer ---- */
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

/* ---- context card ---- */
.ai-context-banner {
    background: linear-gradient(135deg, rgba(27,138,58,0.18), rgba(124,179,66,0.1));
    border: 1px solid rgba(76,175,80,0.4);
    border-radius: 18px;
    padding: 14px 18px;
    margin: 10px 0 16px 0;
}

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


# --------------------------------------------------------------------------- Password Gate
def get_password() -> str:
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
                time.sleep(1)
                st.error("Wrong password. Please try again.")
    return False


if not check_password():
    st.stop()


# --------------------------------------------------------------------------- Model & Helpers
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


def pretty(class_name: str) -> str:
    crop, _, cond = class_name.partition("___")
    crop = crop.replace("_", " ").replace(",", "")
    cond = cond.replace("_", " ").strip()
    return f"{crop} - {cond}"


def crop_of(class_name: str) -> str:
    return class_name.split("___")[0].split("_")[0].split(",")[0]


def generate_in_app_ai_response(user_query: str, diagnosis_ctx: dict, ollama_config: tuple = None) -> str:
    """In-App AI engine running directly inside app backend without external redirection."""
    dis_title = diagnosis_ctx.get("disease", "Unknown Condition") if diagnosis_ctx else "General Query"
    dis_cause = diagnosis_ctx.get("cause", "") if diagnosis_ctx else ""

    ctx_text = "No leaf diagnosis loaded."
    if diagnosis_ctx:
        ctx_text = (
            f"Condition: {dis_title}\n"
            f"Confidence: {diagnosis_ctx.get('confidence', 0):.1%}\n"
            f"Cause: {dis_cause}\n"
            f"Symptoms: {', '.join(diagnosis_ctx.get('symptoms', []))}\n"
            f"Organic Remedies: {', '.join(diagnosis_ctx.get('cultural', []))}\n"
            f"Chemical Treatments: {', '.join(diagnosis_ctx.get('chemical', []))}"
        )

    # Option A: Local Ollama backend connection if enabled
    if ollama_config and ollama_config[0]:
        _, model_name, host = ollama_config
        try:
            payload = json.dumps({
                "model": model_name,
                "stream": False,
                "messages": [
                    {"role": "system", "content": "You are a helpful plant health assistant. Answer practical gardening questions briefly and clearly."},
                    {"role": "user", "content": f"Diagnosis Data:\n{ctx_text}\n\nUser Question: {user_query}"}
                ]
            }).encode("utf-8")
            req = urllib.request.Request(f"{host.rstrip('/')}/api/chat", data=payload, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())["message"]["content"]
        except Exception:
            pass

    # Option B: Built-In Direct Backend AI Engine (Runs inside Streamlit without external pages)
    q_lower = user_query.lower()
    if "eat" in q_lower or "harvest" in q_lower or "safe" in q_lower:
        if diagnosis_ctx and diagnosis_ctx.get('is_healthy'):
            return "✅ **Yes!** Your plant is completely healthy, so harvested produce is safe to eat."
        return "⚠️ **Harvest Safety:** Remove severely affected leaves or infected fruit. Healthy fruit from the rest of the plant is usually safe to eat after thorough washing, but avoid eating fruit showing soft rot or dark lesions."

    if "organic" in q_lower or "home" in q_lower or "natural" in q_lower or "remedy" in q_lower:
        if diagnosis_ctx and diagnosis_ctx.get('cultural'):
            remedies = "\n- ".join(diagnosis_ctx['cultural'])
            return f"🌿 **Organic Care Plan for {dis_title}:**\n\n- {remedies}\n\n*Pro Tip:* Always irrigate at soil level to keep leaves dry and prevent fungal spores from spreading."
        return "🌿 **Organic Care Tips:** Prune low-hanging foliage, apply neem oil or copper fungicide spray, and ensure generous spacing between plants for airflow."

    if "chemical" in q_lower or "medicine" in q_lower or "spray" in q_lower:
        if diagnosis_ctx and diagnosis_ctx.get('chemical'):
            chems = "\n- ".join(diagnosis_ctx['chemical'])
            return f"💊 **Commercial Treatment Options for {dis_title}:**\n\n- {chems}\n\n⚠️ *Always follow safety instructions on the product label.*"
        return "💊 Consult local agricultural centers for fungicides or sprays registered in your area."

    if diagnosis_ctx:
        return (
            f"🤖 **Assistant Analysis for {dis_title}:**\n\n"
            f"Regarding your query: *\"{user_query}\"*\n\n"
            f"- **Cause:** {dis_cause}\n"
            f"- **Next Steps:** Prune affected lower foliage, keep soil moist but well-drained, and sanitize garden tools after trimming.\n\n"
            f"Feel free to ask more details on organic spraying, watering schedules, or prevention!"
        )

    return "🤖 **Plant AI:** I am ready to help! Upload a leaf photo in the **Leaf Scanner** screen to get a tailored analysis."


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
    # --------------------------------------------------------------------------- Sidebar & View Switcher
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
        mode_label = st.selectbox(
            "Image scaling used in training", list(PREPROCESS_MODES),
            help="Must match how the model was trained. If healthy leaves look wrong, try another option.")
    use_llm = st.toggle("Local Ollama Server", value=False,
                        help="Optional. Connects to your local Ollama server if installed.")
    if use_llm:
        llm_model = st.text_input("Ollama model", "llama3.2")
        llm_host = st.text_input("Ollama address", "http://localhost:11434")
    else:
        llm_model, llm_host = None, None

    st.divider()
    if st.button("🚪 Log out"):
        st.session_state["auth_ok"] = False
        st.rerun()

mode = PREPROCESS_MODES[mode_label]
ollama_cfg = (use_llm, llm_model, llm_host)

# =========================================================================== SCREEN 1: LEAF SCANNER
if st.session_state.view == "scanner":
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
        st.markdown('<div class="foot">Decision aid only - confirm serious outbreaks with local experts.</div>', unsafe_allow_html=True)
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
        st.error("The model could not run due to a library version mismatch.")
        st.code(f"{type(exc).__name__}: {exc}")
        st.info("Fix: `pip install -U \"tensorflow>=2.18\"`, then restart the app.")
        st.stop()

    if len(probs) != len(CLASS_NAMES):
        st.error(f"Model has {len(probs)} outputs but app expects {len(CLASS_NAMES)} classes.")
        st.stop()

    order = np.argsort(probs)[::-1]
    best = int(order[0])
    best_name, conf = CLASS_NAMES[best], probs[best]
    info = DISEASE_INFO.get(best_name)

    supported = crop_of(best_name) in SUPPORTED_CROPS and info is not None
    confident = conf >= threshold
    severity = info["severity"] if supported else "none"
    is_healthy = supported and severity == "none"

    # Save diagnosis data directly to session memory for the in-app AI
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
                        '<p>No disease found on this leaf.</p></div></div>',
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

        # Quick In-App AI Chat launch button
        if supported and confident:
            if st.button("💬 Ask AI About This Diagnosis", use_container_width=True):
                st.session_state.view = "ai_assistant"
                st.rerun()

    with st.expander("📊 Top 3 possibilities"):
        top = pd.DataFrame({"Class": [pretty(CLASS_NAMES[i]) for i in order[:3]],
                            "Probability": [probs[i] for i in order[:3]]}).set_index("Class")
        st.bar_chart(top, horizontal=True)

    if not supported:
        st.warning("This app currently covers tomato and potato leaves.")
        st.stop()
    if not confident:
        st.warning("The model is not confident about this photo. Try taking a closer picture in good light.")
        st.stop()

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
    with t_sym:
        bullets(info["symptoms"])
    with t_prev:
        bullets(info["prevention"])

    st.markdown('<div class="foot">Decision aid only - confirm serious outbreaks with local experts.</div>', unsafe_allow_html=True)


# =========================================================================== SCREEN 2: IN-APP AI ASSISTANT
elif st.session_state.view == "ai_assistant":
    st.markdown("## 🤖 In-App Plant AI Assistant")
    st.caption("Ask questions, get customized organic advice, or learn how to care for your plant.")

    curr_diag = st.session_state.current_diagnosis
    if curr_diag:
        st.markdown(
            f'<div class="ai-context-banner">'
            f'<b>📌 Active Diagnosis Context:</b> {curr_diag["disease"]} '
            f'({curr_diag["confidence"]:.1%} confidence)<br>'
            f'<small>The AI is configured with your current leaf diagnosis details.</small>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("💡 No leaf scan loaded yet. You can ask general gardening questions or scan a leaf image first.")

    # Quick action prompt buttons
    st.write("**Quick Questions:**")
    qcol1, qcol2, qcol3 = st.columns(3)
    quick_prompt = None
    with qcol1:
        if st.button("🌿 Organic Remedies"):
            quick_prompt = "What organic remedies should I apply?"
    with qcol2:
        if st.button("⚠️ Safe to Harvest?"):
            quick_prompt = "Is produce from this plant safe to eat?"
    with qcol3:
        if st.button("💊 Spray Precautions"):
            quick_prompt = "What chemical or spray precautions should I follow?"

    st.divider()

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input box
    user_input = st.chat_input("Ask any question about plant care, treatments, or disease prevention...")
    active_query = user_input or quick_prompt

    if active_query:
        # Append user message
        st.session_state.chat_messages.append({"role": "user", "content": active_query})
        with st.chat_message("user"):
            st.markdown(active_query)

        # Generate in-app response directly
        with st.chat_message("assistant"):
            with st.spinner("Analyzing context & generating response..."):
                response = generate_in_app_ai_response(active_query, curr_diag, ollama_cfg)
                st.markdown(response)

        st.session_state.chat_messages.append({"role": "assistant", "content": response})

    st.divider()
    if st.button("⬅️ Back to Leaf Scanner"):
        st.session_state.view = "scanner"
        st.rerun()
