import pandas as pd
import pickle
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ===== Step 1: Load Data =====
train_df = pd.read_csv("Training.csv")
test_df = pd.read_csv("Testing.csv")

# Drop unnamed columns if any
train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

# Separate features and target
X_train = train_df.drop(columns=['prognosis'])
y_train = train_df['prognosis']

X_test = test_df.drop(columns=['prognosis'])
y_test = test_df['prognosis']

# Convert symptoms to binary numeric (0/1)
X_train = X_train.applymap(lambda x: 1 if x != 0 else 0)
X_test = X_test.applymap(lambda x: 1 if x != 0 else 0)

# ===== Step 2: Encode Target Labels =====
label_encoder = LabelEncoder()
y_train_enc = label_encoder.fit_transform(y_train)
y_test_enc = label_encoder.transform(y_test)

# ===== Step 3: Train Model =====
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train_enc)

# ===== Step 4: Evaluate =====
y_pred = model.predict(X_test)
acc = accuracy_score(y_test_enc, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%")

# Save model and encoder
pickle.dump(model, open("disease_model.pkl", "wb"))
pickle.dump(label_encoder, open("label_encoder.pkl", "wb"))

# ===== Step 5: Medicine Recommendation Dictionary =====
medicine_dict = {
    "Fungal infection": ["Fluconazole 150mg – once daily for 3 days", "Apply Clotrimazole cream twice daily"],
    "Allergy": ["Cetirizine 10mg – once daily at night", "Loratadine 10mg – once daily"],
    "Diabetes": ["Metformin 500mg – twice daily after meals", "Glimepiride 2mg – once daily before breakfast"],
    "Hypertension": ["Amlodipine 5mg – once daily", "Losartan 50mg – once daily"],
    "Migraine": ["Sumatriptan 50mg – at migraine onset", "Paracetamol 500mg – as needed"],
}

# ===== Step 6: Prediction Function =====
def predict_disease(symptom_vector):
    loaded_model = pickle.load(open("disease_model.pkl", "rb"))
    loaded_encoder = pickle.load(open("label_encoder.pkl", "rb"))

    prediction_encoded = loaded_model.predict([symptom_vector])[0]
    predicted_disease = loaded_encoder.inverse_transform([prediction_encoded])[0]

    medicines = medicine_dict.get(predicted_disease, ["No medicine found. Consult a doctor."])
    return predicted_disease, medicines

# ===== Step 7: Random Test Case =====
if __name__ == "__main__":
    random_index = random.randint(0, len(X_test) - 1)
    example_symptoms = X_test.iloc[random_index].tolist()
    actual_disease = y_test.iloc[random_index]

    print("\n--- Random Test Case ---")
    print(f"Actual Disease from Dataset: {actual_disease}")

    predicted_disease, meds = predict_disease(example_symptoms)

    print("Predicted Disease:", predicted_disease)
    print("Recommended Medicines:")
    for med in meds:
        print("-", med)
