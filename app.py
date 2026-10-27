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

# Detailed prescription database with store-bought chemical options and homemade remedies
DETAILED_SOLUTIONS = {
    "Tomato___Early_blight": {
        "title": "Tomato Early Blight",
        "action_advice": "Prune infected lower leaves immediately, avoid overhead watering, and apply treatments every 7–10 days.",
        "commercial": "Chlorothalonil 72.5% SC or Mancozeb 75% WP. Mix 2.5 grams per 1 liter of clean water and spray thoroughly over foliage.",
        "organic": "Baking Soda Spray: Mix 1 tablespoon baking soda, 1 teaspoon liquid dish soap, and 1 gallon (3.7L) of water. Spray weekly.",
        "expert": "Consult your local agricultural extension service if more than 30% of the plant shows stem lesions."
    },
    "Tomato___Late_blight": {
        "title": "Tomato Late Blight",
        "action_advice": "Acts extremely fast in wet weather. Remove and destroy heavily infected plants entirely to save the rest of your crop.",
        "commercial": "Copper Hydroxide or Metalaxyl-based fungicides. Mix according to product label instructions and apply preventatively.",
        "organic": "Neem Oil Spray: Mix 2 tablespoons of pure cold-pressed neem oil with 1 teaspoon mild soap in 1 liter of water; spray every 5 days.",
        "expert": "Highly contagious. Contact your local plant protection office immediately for regional containment advice."
    },
    "Tomato___Leaf_Mold": {
        "title": "Tomato Leaf Mold",
        "action_advice": "Improve greenhouse or garden ventilation, space plants wider apart, and reduce high humidity.",
        "commercial": "Copper-based fungicides or Difenoconazole spray. Follow label dosage guidelines strictly.",
        "organic": "Milk Spray: Mix 1 part milk with 9 parts water and spray leaves in full sunlight every 7 days.",
        "expert": "Consult local horticulture experts if cultivating in enclosed greenhouses."
    },
    "Tomato___Septoria_leaf_spot": {
        "title": "Septoria Leaf Spot",
        "action_advice": "Remove lower spotted leaves, clear weed hosts around the garden, and clean garden tools after pruning.",
        "commercial": "Mancozeb or Copper oxychloride fungicide sprays applied every 7 to 10 days during wet spells.",
        "organic": "Garlic & Chilli Spray: Blend garlic cloves and hot peppers with water, strain, and spray over leaves as a natural deterrent.",
        "expert": "Seek local extension advice if crop rotation is needed for next season."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "title": "Spider Mites",
        "action_advice": "Spray strong streams of cold water under the leaves to knock down mite populations physically.",
        "commercial": "Abamectin or horticultural mineral oils. Follow exact label guidelines for dilution.",
        "organic": "Insecticidal Soap or Neem Oil applied thoroughly to the underside of all leaves.",
        "expert": "Consult local experts if infestation overwhelms natural predator populations."
    },
    "Tomato___Target_Spot": {
        "title": "Target Spot",
        "action_advice": "Avoid overhead sprinkler irrigation and ensure proper sunlight penetration through canopy pruning.",
        "commercial": "Azoxystrobin or Chlorothalonil sprays applied at 7-to-14-day intervals.",
        "organic": "Compost tea foliar spray to strengthen natural plant microbial defenses.",
        "expert": "Contact agricultural extension agents for regional fungicide resistance updates."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "title": "Tomato Yellow Leaf Curl Virus",
        "action_advice": "Incurable virus spread by whiteflies. Remove and destroy infected plants immediately to protect neighbors.",
        "commercial": "Imidacloprid or Acetamiprid insecticides targeted specifically at controlling whitefly vectors.",
        "organic": "Yellow sticky traps placed around plants to capture whiteflies, combined with reflective mulches.",
        "expert": "Consult local agricultural authorities regarding resistant hybrid seed varieties for future planting."
    },
    "Tomato___Tomato_mosaic_virus": {
        "title": "Tomato Mosaic Virus",
        "action_advice": "Wash hands thoroughly and sterilize tools after touching infected plants. Do not smoke tobacco near tomatoes.",
        "commercial": "No chemical cure available. Focus strictly on sanitation and virus-free certified seeds.",
        "organic": "Skim milk solution applied to hands and tools before handling plants to inhibit virus transmission.",
        "expert": "Contact a local plant pathology lab for tissue testing if large fields are affected."
    },
    "Tomato___Bacterial_spot": {
        "title": "Bacterial Spot",
        "action_advice": "Avoid working with wet plants; remove crop debris completely at the end of the season.",
        "commercial": "Copper bactericides combined with Mancozeb applied early in the infection cycle.",
        "organic": "Copper soap spray or hydrogen peroxide-based organic bactericide sprays.",
        "expert": "Consult local agricultural extension officers for certified disease-free seed sourcing."
    },
    "Potato___Early_blight": {
        "title": "Potato Early Blight",
        "action_advice": "Ensure balanced nitrogen fertilization and harvest tubers carefully without skin bruising.",
        "commercial": "Chlorothalonil or Mancozeb sprays applied every 7–10 days.",
        "organic": "Neem oil or copper soap sprays applied during dry morning hours.",
        "expert": "Consult local extension advisors for regional potato blight forecasting alerts."
    },
    "Potato___Late_blight": {
        "title": "Potato Late Blight",
        "action_advice": "Destroy potato haulms (vines) 2 weeks before harvest to prevent tuber infection.",
        "commercial": "Cymoxanil or Metalaxyl systemic fungicides according to local agricultural guidelines.",
        "organic": "Copper-based organic sprays applied preventatively before rain events.",
        "expert": "Notify local agricultural authorities immediately as potato late blight spreads aggressively across regions."
    },
    "Healthy": {
        "title": "Healthy Plant",
        "action_advice": "Your plant is in great condition! No treatment is required.",
        "commercial": "None needed. Continue regular watering at the base and balanced fertilization.",
        "organic": "Maintain preventive care with compost and proper spacing.",
        "expert": "Keep up the excellent gardening practices!"
    }
}

st.set_page_config(page_title="Plant Disease Detection AI", page_icon="🌿", layout="centered",
                    initial_sidebar_state="collapsed")


# --------------------------------------------------------------------------- model
@st.cache_resource(show_spinner="Loading AI model (downloading if needed)...")
def load_model():
    import tensorflow as tf  # imported lazily so page loads instantly

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


def bullets(items):
    for item in items:
        st.markdown(f"- {item}")


# ------------------------------------------------------------------ vibrant styling
CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 860px;}
.hero {background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #43a047 100%); color: #fff;
       border-radius: 24px; padding: 32px 24px; text-align: center; margin-bottom: 22px;
       box-shadow: 0 8px 20px rgba(46,125,50,0.25);}
.hero h1 {color: #fff; margin: 0; font-size: 2.3rem; font-weight: 800; letter-spacing: .4px;}
.hero p {color: #e8f5e9; margin: 8px 0 0; font-size: 1.08rem;}
.crops {display: flex; gap: 16px; margin-bottom: 24px;}
.crop {flex: 1; border-radius: 18px; padding: 16px 12px; text-align: center;
       background: #ffffff; border: 2px solid #c8e6c9; box-shadow: 0 4px 12px rgba(0,0,0,0.04);}
.crop svg, .crop img {width: 90px; height: 90px; object-fit: cover; border-radius: 14px;}
.crop b {display: block; margin-top: 8px; font-size: 1.1rem; color: #1b5e20;}
.crop span {font-size: .88rem; color: #558b2f;}
.banner {border-radius: 16px; padding: 16px 20px; margin: 14px 0 18px; color: #fff;
         font-weight: 700; font-size: 1.15rem; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
.b-none {background: #2e7d32;} .b-moderate {background: #f57c00;}
.b-high {background: #e53935;} .b-incurable {background: #b71c1c;}
.solution-box {background: #f1f8e9; border: 2px solid #aed581; border-radius: 16px; padding: 20px; margin: 16px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.03);}
.solution-box h3 {color: #2e7d32; margin-top: 0; font-size: 1.25rem;}
.stat {text-align: center; border-radius: 16px; padding: 14px;
       background: #ffffff; border: 2px solid #c8e6c9; box-shadow: 0 4px 10px rgba(0,0,0,0.03);}
.stat small {color: #558b2f; display: block; font-weight: 600;} .stat b {font-size: 1.3rem; color: #1b5e20;}
.foot {text-align: center; color: #7f8c8d; font-size: .85rem; margin-top: 32px;}
</style>
"""

TOMATO_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
<circle cx="50" cy="56" r="34" fill="#e53935"/><ellipse cx="38" cy="42" rx="9" ry="5" fill="#ff8a80" opacity=".6" transform="rotate(-30 38 42)"/>
<path d="M50 24 L43 14 L50 19 L57 14 Z M50 24 L33 24 L44 30 Z M50 24 L67 24 L56 30 Z M50 24 L38 34 L48 30 Z M50 24 L62 34 L52 30 z"
 fill="#2e7d32"/><rect x="48" y="10" width="4" height="12" rx="2" fill="#1b5e20"/></svg>"""

POTATO_SVG = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
<path d="M18 58 C14 36 34 22 56 26 C78 30 90 46 82 66 C74 84 40 88 26 76 C21 72 19 66 18 58 Z" fill="#c8923f"/>
<path d="M30 40 C40 30 58 30 70 38" stroke="#e6b868" stroke-width="4" fill="none" stroke-linecap="round" opacity=".7"/>
<circle cx="40" cy="58" r="2.6" fill="#8d5a1e"/><circle cx="58" cy="52" r="2.6" fill="#8d5a1e"/>
<circle cx="66" cy="68" r="2.6" fill="#8d5a1e"/><circle cx="46" cy="72" r="2.6" fill="#8d5a1e"/>
<circle cx="30" cy="62" r="2.2" fill="#8d5a1e"/></svg>"""


def crop_visual(name: str, svg: str) -> str:
    for ext in ("jpg", "jpeg", "png", "webp"):
        p = BASE_DIR / "images" / f"{name}.{ext}"
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            return f'<img src="data:image/{ext};base64,{b64}">'
    return svg


def hero_and_crops():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><h1>🌿 Plant Disease Diagnostic AI</h1>'
        '<p>Upload a leaf photo for instant AI diagnosis, store-bought chemical remedies, and organic care</p></div>',
        unsafe_allow_html=True)
    st.markdown(
        f'<div class="crops">'
        f'<div class="crop">{crop_visual("tomato", TOMATO_SVG)}<b>Tomato</b><span>9 Diseases + Healthy</span></div>'
        f'<div class="crop">{crop_visual("potato", POTATO_SVG)}<b>Potato</b><span>2 Diseases + Healthy</span></div>'
        f'</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider("Minimum confidence", 40, 95, 70, 5, format="%d%%") / 100
    with st.expander("Advanced"):
        mode_label = st.selectbox(
            "Image scaling used in training", list(PREPROCESS_MODES),
            help="Must match how the model was trained.")

mode = PREPROCESS_MODES[mode_label]

# ------------------------------------------------------------------ main page
hero_and_crops()

tab_up, tab_cam = st.tabs(["📁 Upload Photo", "📷 Use Camera"])
with tab_up:
    uploaded = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png", "webp"],
                                label_visibility="collapsed")
with tab_cam:
    camera = st.camera_input("Take a photo of the leaf", label_visibility="collapsed")

source = uploaded or camera
if source is None:
    st.info("💡 **Tip:** Take a close, well-lit photo of ONE leaf on a plain background for best accuracy.")
    st.markdown('<div class="foot">Plant Disease Diagnostic System • Designed for Farmers & Gardeners</div>', unsafe_allow_html=True)
    st.stop()

image_bytes = source.getvalue()
try:
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
except Exception:
    st.error("That file could not be read as an image. Please try another photo.")
    st.stop()

try:
    with st.spinner("Analyzing leaf..."):
        probs = predict(image_bytes, mode)
except Exception as exc:
    st.error("The model could not run.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.stop()

order = np.argsort(probs)[::-1]
best = int(order[0])
best_name, conf = CLASS_NAMES[best], probs[best]
info = DISEASE_INFO.get(best_name, {})

col_img, col_res = st.columns([1, 1.2], gap="large")
with col_img:
    st.image(image, caption="Your Leaf Sample", use_column_width=True)
with col_res:
    st.markdown(f'<div class="stat"><small>Diagnostic Prediction</small><b>{pretty(best_name)}</b></div>',
                unsafe_allow_html=True)
    st.markdown(f'<div class="stat" style="margin-top:12px"><small>Confidence Score</small>'
                f'<b>{conf:.1%}</b></div>', unsafe_allow_html=True)
    st.progress(min(max(conf, 0.0), 1.0))

if crop_of(best_name) not in SUPPORTED_CROPS:
    st.warning(f"This appears to be **{pretty(best_name)}**, which this app does not cover. Please upload a tomato or potato leaf.")
    st.stop()
if conf < threshold:
    st.warning("The model is **not confident** enough about this photo. Please retake it closer, in daylight, with one leaf filling most of the frame.")
    st.stop()

# -------- Results & Solutions Section directly below
severity = info.get("severity", "moderate")
label = {"none": "✅ Your plant is HEALTHY! No treatment needed.",
         "moderate": f"⚠️ Disease Detected: {info.get('title', best_name)} - Treat promptly.",
         "high": f"🚨 Severe Disease: {info.get('title', best_name)} - Spreads fast, act today!",
         "incurable": f"🛑 Critical Issue: {info.get('title', best_name)} - Protect surrounding crops."}[severity]

st.markdown(f'<div class="banner b-{severity}">{label}</div>', unsafe_allow_html=True)

# Fetch detailed custom solutions
sol = DETAILED_SOLUTIONS.get(best_name, {
    "title": pretty(best_name),
    "action_advice": "Isolate the infected plant parts and maintain good airflow.",
    "commercial": "Consult your local agricultural store for registered broad-spectrum fungicides.",
    "organic": "Apply organic neem oil spray weekly as a preventive measure.",
    "expert": "Consult your local agricultural extension service for specialized advice."
})

st.markdown(f"""
<div class="solution-box">
    <h3>📋 Action Plan & Solutions</h3>
    <p><b>Recommended Action:</b> {sol['action_advice']}</p>
    <hr style="border: 0; border-top: 1px solid #c8e6c9; margin: 12px 0;">
    <p><b>💊 Store-Bought Chemical Treatment:</b><br>{sol['commercial']}</p>
    <p><b>🌿 Homemade / Organic Remedy:</b><br>{sol['organic']}</p>
    <p><b>👨‍🌾 Expert Consultation:</b><br>{sol['expert']}</p>
</div>
""", unsafe_allow_html=True)

# Tabs for additional details
t_sym, t_prev = st.tabs(["🔍 Detailed Symptoms", "🛡️ Prevention Tips"])
with t_sym:
    if "symptoms" in info:
        bullets(info["symptoms"])
with t_prev:
    if "prevention" in info:
        bullets(info["prevention"])

st.markdown('<div class="foot">Plant Disease Diagnostic System • Built for accurate crop protection</div>', unsafe_allow_html=True)
