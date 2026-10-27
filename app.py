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
        "action_advice": "Ensure balanced nitrogen fertilization and harvest
