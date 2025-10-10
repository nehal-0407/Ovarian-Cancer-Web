# 🧬 Ovarian Cancer Prediction Web App

This is a **Streamlit-based** web application that predicts ovarian cancer outcomes using a pre-trained hybrid model stored as `hybrid_models.pkl`.

## 🧩 Folder Structure
```
ovarian_cancer_app_with_hybrid/
├── app.py
├── hybrid_models.pkl
├── requirements.txt
└── README.md
```

## ⚙️ How to Run Locally

1. Clone this project or copy the folder.
2. Make sure Python 3.8+ is installed.
3. Open terminal in this folder and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```
5. The app will open automatically in your browser at `http://localhost:8501`.

## ☁️ Deployment (Streamlit Cloud)

1. Push these files to a **GitHub repository**.
2. Go to [https://share.streamlit.io](https://share.streamlit.io).
3. Click **New app** → select your repo and `app.py`.
4. Streamlit Cloud will auto-deploy your app!

## ✅ Notes
- Ensure your model file is named `hybrid_models.pkl`.
- The app will prompt you to enter the same number of input features used during model training.
