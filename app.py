from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ------------------------------
# 1) Load data & train the model
# ------------------------------
def load_data():
    train_df = pd.read_csv("Training.csv")
    # Drop stray "Unnamed" columns sometimes created by Excel exports
    train_df = train_df.loc[:, ~train_df.columns.str.contains(r"^Unnamed")]
    return train_df

def get_accuracy(model, X_train, y_train):
    # Prefer Testing.csv if available, else use hold-out split on training
    if os.path.exists("Testing.csv"):
        test_df = pd.read_csv("Testing.csv")
        test_df = test_df.loc[:, ~test_df.columns.str.contains(r"^Unnamed")]
        X_test = test_df.drop(columns=["prognosis"])
        y_test = test_df["prognosis"]
        return accuracy_score(y_test, model.predict(X_test))
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
        tmp_model = RandomForestClassifier(n_estimators=200, random_state=42)
        tmp_model.fit(X_tr, y_tr)
        return accuracy_score(y_te, tmp_model.predict(X_te))

train_df = load_data()
X = train_df.drop(columns=["prognosis"])
y = train_df["prognosis"]

clf = RandomForestClassifier(n_estimators=300, random_state=42)
clf.fit(X, y)

symptoms_list = X.columns.tolist()
model_accuracy = get_accuracy(clf, X, y)

print(f"✅ Disease model ready. Accuracy: {model_accuracy*100:.2f}%")

# ---------------------------------------------------------
# 2) Ensure medicines.csv exists (cover ALL diseases in y)
# ---------------------------------------------------------
MEDS_CSV = "medicines.csv"

# Base safe, OTC-style suggestions + general-care advice
# (We’ll map what we can; everything else gets a safe fallback)
BASE_RECS = {
    "Fungal infection": [
        "Topical antifungal cream (clotrimazole 1%)",
        "Keep area clean and dry",
        "Wear loose, breathable clothing"
    ],
    "Allergy": [
        "Non-drowsy antihistamine (cetirizine/loratadine) per label",
        "Saline nasal rinse",
        "Avoid known triggers"
    ],
    "GERD": [
        "Antacid/alginate as per label",
        "Small meals; avoid late-night eating",
        "Elevate head while sleeping"
    ],
    "Peptic ulcer disease": [
        "Antacid for symptom relief",
        "Avoid NSAIDs and spicy/fatty foods",
        "See a clinician for evaluation"
    ],
    "Gastroenteritis": [
        "Oral Rehydration Solution (ORS)",
        "Light foods as tolerated",
        "Wash hands; rest"
    ],
    "Common Cold": [
        "Rest and fluids",
        "Acetaminophen (paracetamol) per label for fever/pain",
        "Saline nasal spray/lozenges"
    ],
    "Migraine": [
        "Acetaminophen or ibuprofen per label at onset",
        "Hydration; dark quiet room",
        "Limit screen exposure"
    ],
    "Hypertension": [
        "Reduce salt; regular light exercise if safe",
        "Home BP monitoring",
        "Consult a clinician (no OTC meds recommended)"
    ],
    "Diabetes": [
        "Monitor blood sugar if available",
        "Balanced meals; hydrate",
        "Consult a clinician (avoid self-medication)"
    ],
    "Hypoglycemia": [
        "Take fast-acting carbs (e.g., glucose tablets/juice)",
        "Re-check glucose after 15 minutes",
        "Seek medical help if unresolved"
    ],
    "Hypothyroidism": [
        "Adequate rest; balanced diet",
        "Warm clothing if cold-intolerant",
        "Follow-up with clinician for management"
    ],
    "Hyperthyroidism": [
        "Hydrate; avoid excess caffeine",
        "Light meals",
        "See clinician for evaluation"
    ],
    "Osteoarthritis": [
        "Topical pain-relief gel per label",
        "Heat/ice as needed",
        "Gentle range-of-motion exercises"
    ],
    "Psoriasis": [
        "Unscented moisturizers",
        "Coal tar or salicylic acid shampoo for scalp (per label)",
        "Avoid harsh soaps; hydrate"
    ],
    "Impetigo": [
        "Keep lesions clean and covered",
        "Avoid scratching; wash hands",
        "Seek clinical care (often requires antibiotics)"
    ],
    "Pneumonia": [
        "Urgent clinical care recommended",
        "Rest and fluids",
        "Acetaminophen for fever per label"
    ],
    "Bronchial Asthma": [
        "Avoid triggers; monitor symptoms",
        "Pursed-lip breathing",
        "Seek clinician guidance; do not self-medicate"
    ],
    "Chicken pox": [
        "Calamine lotion/oatmeal baths for itch",
        "Acetaminophen per label (avoid ibuprofen unless advised)",
        "Keep nails short to reduce scratching"
    ],
    "Dengue": [
        "Fluids/ORS; rest",
        "Acetaminophen per label (avoid ibuprofen/aspirin)",
        "Urgent clinical care if bleeding or severe pain"
    ],
    "Malaria": [
        "Urgent clinical care recommended",
        "Hydration and rest",
        "Use bed nets/repellent to prevent bites"
    ],
    "Typhoid": [
        "Seek medical care promptly",
        "ORS and hydration",
        "Eat light, safe foods"
    ],
    "Jaundice": [
        "Avoid alcohol and unnecessary medicines",
        "Hydrate well",
        "Seek clinical evaluation"
    ],
    "Tuberculosis": [
        "Seek clinical care promptly",
        "Good ventilation; cover cough",
        "Balanced nutrition and rest"
    ],
    "Alcoholic hepatitis": [
        "Stop alcohol completely",
        "Hydration; balanced diet",
        "Seek clinical care"
    ],
    "Dimorphic hemmorhoids(piles)": [
        "Warm sitz baths",
        "High-fiber diet + fluids",
        "OTC hemorrhoid cream per label"
    ],
    "Varicose veins": [
        "Leg elevation",
        "Light activity; avoid prolonged standing",
        "Consider compression socks (ask clinician first)"
    ],
    "Cervical spondylosis": [
        "Heat therapy; gentle neck stretches",
        "Ergonomic posture",
        "Topical analgesic per label"
    ],
    "Paralysis (brain hemorrhage)": [
        "Emergency care immediately",
        "Do not self-medicate",
        "Keep patient safe while awaiting help"
    ],
    "Heart attack": [
        "Call emergency services immediately",
        "Stay still and calm",
        "Do not drive yourself"
    ],
    "Hepatitis A": [
        "Hydrate; rest",
        "Avoid alcohol",
        "Seek medical care"
    ],
    "Hepatitis B": [
        "Avoid alcohol",
        "Seek medical evaluation",
        "Hydrate; balanced diet"
    ],
    "Hepatitis C": [
        "Avoid alcohol",
        "Seek medical care",
        "Hydration and rest"
    ],
    "Hepatitis D": [
        "Avoid alcohol",
        "Seek medical evaluation",
        "Hydrate; rest"
    ],
    "Hepatitis E": [
        "ORS and safe fluids",
        "Avoid alcohol",
        "Seek medical care (esp. pregnancy)"
    ],
    "AIDS": [
        "Seek specialist care",
        "Balanced nutrition and rest",
        "Prevent infections (hygiene, safe practices)"
    ],
}

FALLBACK_RECS = [
    "Rest and hydrate well",
    "Acetaminophen (paracetamol) per label for fever/pain if needed",
    "Seek medical advice if symptoms worsen or persist"
]

def ensure_medicines_csv(diseases: pd.Series):
    if os.path.exists(MEDS_CSV):
        return
    rows = []
    for d in sorted(diseases.unique()):
        if d in BASE_RECS:
            recs = BASE_RECS[d]
        else:
            recs = FALLBACK_RECS
        # store as a semicolon-separated string (easy to read/parse)
        rows.append({"Disease": d, "Recommendations": "; ".join(recs)})
    pd.DataFrame(rows).to_csv(MEDS_CSV, index=False)
    print(f"📝 Created {MEDS_CSV} with {len(rows)} rows (one for each disease).")

ensure_medicines_csv(y)

# ---------------------------------------------------
# 3) Load medicines.csv into a mapping for fast lookups
# ---------------------------------------------------
meds_df = pd.read_csv(MEDS_CSV)
meds_map = {row["Disease"]: [s.strip() for s in str(row["Recommendations"]).split(";")] for _, row in meds_df.iterrows()}

# ------------------------------
# 4) Flask routes
# ------------------------------
@app.route("/")
def home():
    return render_template(
        "index.html",
        symptoms=symptoms_list,
        accuracy=f"{model_accuracy*100:.2f}%"
    )

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    selected = data.get("symptoms", [])

    # Build 1/0 vector in the exact column order used for training
    vec = np.array([1 if s in selected else 0 for s in symptoms_list], dtype=int).reshape(1, -1)
    pred = clf.predict(vec)[0]
    recs = meds_map.get(pred, FALLBACK_RECS)

    return jsonify({
        "disease": pred,
        "recommendations": recs
    })

@app.route("/random", methods=["GET"])
def random_predict():
    row = train_df.sample(1)
    present_syms = [symptoms_list[i] for i, v in enumerate(row[symptoms_list].values[0]) if v == 1]
    pred = clf.predict(row[symptoms_list])[0]
    recs = meds_map.get(pred, FALLBACK_RECS)
    return jsonify({
        "symptoms": present_syms,
        "disease": pred,
        "recommendations": recs
    })

# ------------------------------
# 5) Run
# ------------------------------
if __name__ == "__main__":
    # Add a friendly disclaimer in server logs
    print("\n⚠️  Disclaimer: This tool provides general wellness info and OTC-style guidance only.\n"
          "It is not a substitute for professional medical advice, diagnosis, or treatment.\n")
    app.run(debug=True)

