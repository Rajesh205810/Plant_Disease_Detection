"""Class list + treatment knowledge base for the Tomato & Potato Leaf Doctor.

CLASS_NAMES matches your exact custom dataset folder order.
"""

CLASS_NAMES = [
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

SUPPORTED_CROPS = ("Tomato", "Potato")

DISEASE_INFO = {
    "Potato___healthy": {
        "title": "Healthy potato leaf", 
        "severity": "none",
        "cause": "No disease detected.",
        "symptoms": ["Uniform green colour, no spots, lesions or curling."],
        "cultural": ["Keep monitoring weekly.", "Water at the base of the plant."],
        "chemical": [],
        "prevention": ["Plant certified disease-free seed tubers."]
    },
    "Potato___Early_blight": {
        "title": "Potato early blight", 
        "severity": "moderate",
        "cause": "Fungus Alternaria solani.",
        "symptoms": ["Dark brown spots with concentric target rings."],
        "cultural": ["Remove badly spotted lower leaves."],
        "chemical": ["Protectant fungicides: chlorothalonil or mancozeb."],
        "prevention": ["Use 2-3 year crop rotation."]
    },
    "Potato___Late_blight": {
        "title": "Potato late blight", 
        "severity": "high",
        "cause": "Oomycete Phytophthora infestans.",
        "symptoms": ["Greasy, grey-green blotches turning brown-black."],
        "cultural": ["Destroy infected plants immediately."],
        "chemical": ["Protectant fungicides like chlorothalonil."],
        "prevention": ["Plant certified blight-free seed."]
    },
    "Tomato___Bacterial_spot": {
        "title": "Tomato bacterial spot", 
        "severity": "moderate",
        "cause": "Bacteria (Xanthomonas species).",
        "symptoms": ["Small, dark, water-soaked spots."],
        "cultural": ["Use drip irrigation."],
        "chemical": ["Copper-based bactericide."],
        "prevention": ["Use certified disease-free seed."]
    },
    "Tomato___Early_blight": {
        "title": "Tomato early blight", 
        "severity": "moderate",
        "cause": "Fungus Alternaria solani.",
        "symptoms": ["Brown spots with concentric rings."],
        "cultural": ["Pick off affected lower leaves."],
        "chemical": ["Chlorothalonil or mancozeb."],
        "prevention": ["Rotate crops 2-3 years."]
    },
    "Tomato___Late_blight": {
        "title": "Tomato late blight", 
        "severity": "high",
        "cause": "Oomycete Phytophthora infestans.",
        "symptoms": ["Large, greasy grey-green blotches."],
        "cultural": ["Remove and destroy infected plants."],
        "chemical": ["Protectant fungicides."],
        "prevention": ["Use blight-resistant varieties."]
    },
    "Tomato___Leaf_Mold": {
        "title": "Tomato leaf mold", 
        "severity": "moderate",
        "cause": "Fungus Passalora fulva.",
        "symptoms": ["Pale green patches turning to olive-green mould."],
        "cultural": ["Ventilate and reduce humidity."],
        "chemical": ["Chlorothalonil or copper-based fungicides."],
        "prevention": ["Grow resistant varieties."]
    },
    "Tomato___Septoria_leaf_spot": {
        "title": "Tomato Septoria leaf spot", 
        "severity": "moderate",
        "cause": "Fungus Septoria lycopersici.",
        "symptoms": ["Small round spots with dark borders."],
        "cultural": ["Remove infected lower leaves."],
        "chemical": ["Chlorothalonil or mancozeb."],
        "prevention": ["Rotate 3 years away from tomato."]
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "title": "Tomato spider mites (two-spotted)", 
        "severity": "moderate",
        "cause": "Tetranychus urticae pests.",
        "symptoms": ["Fine yellow stippling and webbing."],
        "cultural": ["Spray leaf undersides with strong water jet."],
        "chemical": ["Insecticidal soap or neem oil."],
        "prevention": ["Scout leaf undersides weekly."]
    },
    "Tomato___Target_Spot": {
        "title": "Tomato target spot", 
        "severity": "moderate",
        "cause": "Fungus Corynespora cassiicola.",
        "symptoms": ["Brown spots with concentric rings and yellow halos."],
        "cultural": ["Prune for airflow."],
        "chemical": ["Chlorothalonil or mancozeb."],
        "prevention": ["Rotate crops and use clean transplants."]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "title": "Tomato yellow leaf curl virus (TYLCV)", 
        "severity": "incurable",
        "cause": "Virus spread by whiteflies.",
        "symptoms": ["Leaves curl upward and turn yellow."],
        "cultural": ["Remove and destroy infected plants early."],
        "chemical": ["Insecticides for whitefly control."],
        "prevention": ["Plant resistant varieties."]
    },
    "Tomato___Tomato_mosaic_virus": {
        "title": "Tomato mosaic virus (ToMV)", 
        "severity": "incurable",
        "cause": "Virus spread by hands, tools, and contact.",
        "symptoms": ["Light/dark green mosaic pattern on leaves."],
        "cultural": ["Remove infected plants; disinfect tools."],
        "chemical": [],
        "prevention": ["Plant certified virus-free seed."]
    },
    "Tomato___healthy": {
        "title": "Healthy tomato leaf", 
        "severity": "none",
        "cause": "No disease detected.",
        "symptoms": ["Even green colour, no spots."],
        "cultural": ["Water at the base."],
        "chemical": [],
        "prevention": ["Stake/prune for airflow."]
    }
}
