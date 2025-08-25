# 🏥 Personalized Health Recommendation System

## 📌 Overview
This project is a **Personalized Health Recommendation System** that predicts diseases based on user symptoms and recommends **safe, general-purpose medicines** along with **general care advice**.  
It combines **Machine Learning (Random Forest)** for disease prediction with a **Flask backend** and **HTML/JS frontend** for user interaction.

⚠️ **Disclaimer:** This system is for **educational purposes only** and does not replace professional medical consultation.

---

## ✨ Features
- ✅ Predicts possible diseases based on symptoms  
- ✅ Suggests safe, non-prescription medicines and self-care advice  
- ✅ User-friendly web interface (checkbox selection of symptoms)  
- ✅ Flask REST API with `/predict` endpoint  
- ✅ Auto-generated `medicines.csv` ensures all diseases have mapped treatments  
- ✅ Logs predictions into `predictions_log.csv` for analysis  

---

## 📂 Project Structure
📦 Health-Recommender-System
┣ 📜 app.py # Flask backend (ML model + API)
┣ 📜 index.html # Frontend (user interface)
┣ 📜 Training.csv # Training dataset (symptoms → disease)
┣ 📜 Testing.csv # Testing dataset
┣ 📜 medicines.csv # Auto-generated medicine recommendations
┣ 📜 predictions_log.csv # Stores user predictions
┣ 📜 README.md # Project documentation

🧠 Machine Learning Model
Algorithm: Random Forest Classifier
Input: Binary symptom indicators
Output: Predicted disease + confidence score

💊 Medicine Recommendation
Each disease is mapped with:
  1)General-purpose medicines (e.g., Paracetamol, Antihistamine).
  2)Self-care advice (e.g., hydration, rest, warm fluids).
Stored in medicines.csv for easy customization.

📊 Results
High accuracy achieved on Testing.csv.

🛠️ Technologies Used
Python, Flask
Scikit-learn, Pandas, NumPy
HTML, CSS, JavaScript

📌 Future Improvements
Add larger medical datasets
Include doctor consultation option
Build a mobile app version
Use deep learning for improved accuracy
