"""Class list + treatment knowledge base for the Tomato & Potato Leaf Doctor.

CLASS_NAMES must be in the SAME ORDER as the folders used when the model was
trained (Keras sorts folder names alphabetically). The list below is the standard
38-class PlantVillage order, which matches the 38-unit output layer of model.h5.
"""

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy",
    "Grape___Black_rot", "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
    "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

SUPPORTED_CROPS = ("Tomato", "Potato")

# severity: "none" | "moderate" | "high" | "incurable"
DISEASE_INFO = {
    "Potato___healthy": {
        "title": "Healthy potato leaf", "severity": "none",
        "cause": "No disease detected.",
        "symptoms": ["Uniform green colour, no spots, lesions or curling."],
        "cultural": ["Keep monitoring weekly, especially after rain or humid spells.",
                     "Water at the base of the plant, in the morning."],
        "chemical": [],
        "prevention": ["Plant certified disease-free seed tubers.",
                       "Rotate away from potato/tomato for 2-3 years.",
                       "Hill soil around stems to protect tubers."],
    },
    "Potato___Early_blight": {
        "title": "Potato early blight", "severity": "moderate",
        "cause": "Fungus Alternaria solani. Favoured by warm weather, alternating wet/dry "
                 "periods and plants stressed by low nitrogen or drought.",
        "symptoms": ["Dark brown spots with concentric 'target' rings, often with a yellow halo.",
                     "Starts on older, lower leaves and moves upward.",
                     "Heavily spotted leaves yellow and drop."],
        "cultural": ["Remove and destroy badly spotted lower leaves.",
                     "Water at the soil level; avoid wetting foliage.",
                     "Fertilise adequately - weak, hungry plants are hit hardest.",
                     "Clear crop debris after harvest."],
        "chemical": ["Protectant fungicides: chlorothalonil, mancozeb or copper-based products.",
                     "Severe cases: products containing azoxystrobin or other strobilurins - "
                     "rotate with a different group to delay resistance.",
                     "Start at first symptoms and repeat on the label interval."],
        "prevention": ["Use 2-3 year crop rotation.", "Choose later-maturing, more tolerant varieties.",
                       "Keep rows spaced for airflow."],
    },
    "Potato___Late_blight": {
        "title": "Potato late blight", "severity": "high",
        "cause": "Oomycete Phytophthora infestans (the cause of the Irish potato famine). "
                 "Spreads fast in cool (about 10-25 C), wet, humid weather; spores travel by wind and "
                 "rain splash and can wipe out a field in days.",
        "symptoms": ["Greasy, grey-green blotches that quickly turn brown-black, often with a pale halo.",
                     "White fuzzy growth on the underside of leaves in humid conditions.",
                     "Dark lesions on stems; tubers develop firm brown rot."],
        "cultural": ["ACT IMMEDIATELY: dig up and destroy infected plants (bag them, burn or bury "
                     "deeply - do not compost).",
                     "Destroy cull piles and volunteer potatoes, which carry the pathogen over winter.",
                     "Avoid overhead irrigation and excess nitrogen.",
                     "Kill vines 2-3 weeks before harvest and harvest in dry weather; discard "
                     "any tubers with rot."],
        "chemical": ["Fungicides PROTECT healthy tissue but cannot cure infected leaves - apply "
                     "before or at the very first signs, with complete coverage.",
                     "Protectants: chlorothalonil, mancozeb, copper. Commercial programmes also use "
                     "systemic/translaminar products (e.g. cymoxanil, mandipropamid) - rotate groups.",
                     "Repeat every 5-10 days while weather stays wet."],
        "prevention": ["Plant certified blight-free seed.", "Rotate out of potato/tomato for 2 years.",
                       "Hill tubers well; choose resistant varieties where available.",
                       "Report outbreaks to your local agricultural extension service."],
    },
    "Tomato___healthy": {
        "title": "Healthy tomato leaf", "severity": "none",
        "cause": "No disease detected.",
        "symptoms": ["Even green colour, no spots, mottling or curling."],
        "cultural": ["Continue weekly scouting (check leaf undersides too).",
                     "Water at the base; mulch to stop soil splash."],
        "chemical": [],
        "prevention": ["Stake/prune for airflow.", "Rotate away from tomato/potato/pepper/eggplant for 2-3 years.",
                       "Buy disease-free transplants."],
    },
    "Tomato___Bacterial_spot": {
        "title": "Tomato bacterial spot", "severity": "moderate",
        "cause": "Bacteria (Xanthomonas species). Spread by rain splash, wet foliage, infected "
                 "seed/transplants and tools. Common in warm, wet weather.",
        "symptoms": ["Small, dark, water-soaked spots that turn brown/black, sometimes with yellow halos.",
                     "Spots may fall out, leaving a 'shot-hole' look.",
                     "Rough, scabby raised spots on fruit."],
        "cultural": ["Remove badly infected leaves; do not work among wet plants.",
                     "Use drip irrigation, not overhead sprinklers.",
                     "Disinfect tools and stakes."],
        "chemical": ["Copper-based bactericide, often tank-mixed with mancozeb, started early and "
                     "repeated on the label interval.",
                     "Copper only protects and slows spread; many bacterial populations are "
                     "partly copper-tolerant, so do not expect a cure."],
        "prevention": ["Use certified disease-free seed and transplants.", "Rotate 2-3 years away from tomato/pepper.",
                       "Remove volunteer tomatoes and crop debris."],
    },
    "Tomato___Early_blight": {
        "title": "Tomato early blight", "severity": "moderate",
        "cause": "Fungus Alternaria solani, favoured by warm, humid weather and stressed plants.",
        "symptoms": ["Brown spots with concentric rings ('bullseye') and yellow halo on older leaves.",
                     "Dark sunken lesions on stems and at the fruit stem end.",
                     "Heavily affected leaves yellow and fall, exposing fruit to sunburn."],
        "cultural": ["Pick off and destroy affected lower leaves.",
                     "Mulch to prevent soil splash; water at the base.",
                     "Keep plants well fed and watered; stake and prune for airflow."],
        "chemical": ["Chlorothalonil, mancozeb or copper-based fungicides, starting at first symptoms.",
                     "Strobilurin (e.g. azoxystrobin) products can be rotated in - follow the label "
                     "interval and limit the number of sprays."],
        "prevention": ["Rotate crops 2-3 years.", "Remove crop debris at season end.",
                       "Use resistant/tolerant varieties and certified transplants."],
    },
    "Tomato___Late_blight": {
        "title": "Tomato late blight", "severity": "high",
        "cause": "Oomycete Phytophthora infestans - the same pathogen as potato late blight. "
                 "Explodes in cool, wet, humid weather.",
        "symptoms": ["Large, greasy grey-green blotches turning brown-black on leaves and stems.",
                     "White fuzzy growth under leaves in humid conditions.",
                     "Firm, brown, greasy-looking patches on fruit."],
        "cultural": ["ACT IMMEDIATELY: remove and destroy infected plants (bag or bury - never compost).",
                     "Check nearby potatoes and tomatoes - the disease moves between them.",
                     "Avoid overhead watering; improve airflow."],
        "chemical": ["Protectant fungicides (chlorothalonil, mancozeb, copper) prevent new infection "
                     "but will not cure diseased tissue; keep coverage continuous in wet weather.",
                     "Rotate fungicide groups per the label to manage resistance."],
        "prevention": ["Use blight-resistant varieties where available.",
                       "Buy healthy transplants; destroy volunteer potatoes/tomatoes.",
                       "Rotate out of tomato/potato for 2 years."],
    },
    "Tomato___Leaf_Mold": {
        "title": "Tomato leaf mold", "severity": "moderate",
        "cause": "Fungus Passalora fulva. Mostly a greenhouse/high-tunnel problem with high humidity "
                 "(above ~85%) and poor ventilation.",
        "symptoms": ["Pale green to yellow patches on the upper leaf surface.",
                     "Olive-green to grey-brown velvety mould on the underside, directly below.",
                     "Leaves curl, wither and drop."],
        "cultural": ["Lower humidity: ventilate, space plants, prune lower leaves.",
                     "Water early in the day, at the base only.",
                     "Remove and destroy infected leaves."],
        "chemical": ["Chlorothalonil or copper-based fungicides when cultural controls are not enough; "
                     "check the label for greenhouse use."],
        "prevention": ["Grow resistant varieties.", "Keep greenhouse humidity low and air moving.",
                       "Sanitise structures between crops."],
    },
    "Tomato___Septoria_leaf_spot": {
        "title": "Tomato Septoria leaf spot", "severity": "moderate",
        "cause": "Fungus Septoria lycopersici. Spread by rain splash from infected debris; likes "
                 "warm, wet weather.",
        "symptoms": ["Many small round spots with dark borders and light grey centres, "
                     "often with tiny black dots (spore bodies) inside.",
                     "Starts on lower leaves after fruit set and moves up.",
                     "Severe cases defoliate the plant."],
        "cultural": ["Remove infected lower leaves; mulch under plants.",
                     "Water at the base; avoid working in wet foliage.",
                     "Clean up all debris at season end."],
        "chemical": ["Chlorothalonil, mancozeb or copper-based fungicides at first symptoms; repeat per label."],
        "prevention": ["Rotate 3 years away from tomato.", "Stake plants for airflow.",
                       "Control nightshade-family weeds."],
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "title": "Tomato spider mites (two-spotted)", "severity": "moderate",
        "cause": "Tiny pests (Tetranychus urticae), not a disease. Explode in hot, dry, dusty conditions.",
        "symptoms": ["Fine yellow stippling on leaves, which turns bronze and dry.",
                     "Fine webbing on leaf undersides and between stems.",
                     "Tiny moving dots under a hand lens."],
        "cultural": ["Spray plants (especially leaf undersides) with a strong jet of water.",
                     "Keep plants well watered; reduce dust.",
                     "Release predatory mites (e.g. Phytoseiulus persimilis) in greenhouses."],
        "chemical": ["Insecticidal soap, horticultural oil or neem oil, with thorough underside coverage, "
                     "repeated every 5-7 days.",
                     "Miticides (e.g. abamectin) only as a last resort and rotated; avoid broad-spectrum "
                     "insecticides that kill the mites' natural enemies."],
        "prevention": ["Scout leaf undersides weekly in hot weather.", "Avoid drought stress.",
                       "Remove weedy hosts near the crop."],
    },
    "Tomato___Target_Spot": {
        "title": "Tomato target spot", "severity": "moderate",
        "cause": "Fungus Corynespora cassiicola. Favoured by warm, humid, wet conditions and dense foliage.",
        "symptoms": ["Brown spots with concentric rings and yellow halos (looks like early blight).",
                     "Spots can also appear on stems and fruit.",
                     "Leaves yellow and drop in severe cases."],
        "cultural": ["Prune for airflow; remove infected leaves.",
                     "Avoid overhead irrigation.",
                     "Remove crop debris after harvest."],
        "chemical": ["Chlorothalonil, mancozeb or strobilurin-containing products per label; rotate groups."],
        "prevention": ["Rotate crops.", "Space and stake plants.", "Use clean transplants."],
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "title": "Tomato yellow leaf curl virus (TYLCV)", "severity": "incurable",
        "cause": "Virus spread by whiteflies (Bemisia tabaci). There is NO cure once a plant is infected.",
        "symptoms": ["Leaves curl upward and turn yellow at the edges; plant is stunted and bushy.",
                     "Flowers drop and fruit set is poor or absent."],
        "cultural": ["Remove and destroy infected plants early (bag them first so whiteflies cannot escape).",
                     "Control whiteflies: yellow sticky traps, reflective mulch, insect-proof netting.",
                     "Remove weeds that harbour whiteflies."],
        "chemical": ["Chemicals do not cure the virus; they only reduce whitefly numbers. "
                     "Use insecticides labelled for whitefly and rotate modes of action to avoid resistance."],
        "prevention": ["Plant TYLCV-resistant/tolerant varieties.", "Use virus-free transplants under netting.",
                       "Avoid planting next to infected or older tomato crops."],
    },
    "Tomato___Tomato_mosaic_virus": {
        "title": "Tomato mosaic virus (ToMV)", "severity": "incurable",
        "cause": "Very stable virus spread by hands, tools, contaminated seed and plant-to-plant contact. "
                 "There is NO cure.",
        "symptoms": ["Light/dark green mottled 'mosaic' pattern on leaves.",
                     "Distorted, fern-like or narrow leaves; stunted growth.",
                     "Uneven ripening and reduced yield."],
        "cultural": ["Remove and destroy infected plants; do not compost.",
                     "Wash hands with soap and water; disinfect tools and stakes (e.g. 10% bleach solution).",
                     "Do not handle tobacco products before touching plants."],
        "chemical": ["No chemical treatment exists. Fungicides and insecticides have no effect."],
        "prevention": ["Plant resistant varieties and certified virus-free seed.",
                       "Rotate crops and clear all plant debris.", "Control weeds that host the virus."],
    },
}
