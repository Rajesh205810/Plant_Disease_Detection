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

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded"
)

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

/* ---- sidebar AI links ---- */
.ai-link-card {
    text-decoration: none !important;
    padding: 10px 14px;
    margin: 6px 0;
    background: rgba(76, 175, 80, 0.12);
    border: 1px solid rgba(76, 175, 80, 0.35);
    border-radius: 14px;
    color: inherit !important;
    font-weight: 600;
    font-size: 0.9rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: all 0.2s ease;
}
.ai-link-card:hover {
    background: rgba(76, 175, 80, 0.25);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.ai-badge {
    font-size: 0.72rem;
    padding: 2px 8px;
    background: rgba(76, 175, 80, 0.25);
    border-radius: 99px;
    opacity: 0.85;
}

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
