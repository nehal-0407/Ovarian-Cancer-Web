import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import pickle
import numpy as np
import os
import gdown
import sys, types

# ---- Patch for missing pytorch_grad_cam ----
if "pytorch_grad_cam" not in sys.modules:
    fake_cam = types.ModuleType("pytorch_grad_cam")
    sys.modules["pytorch_grad_cam"] = fake_cam
    for sub in [
        "activations_and_gradients",
        "base_cam",
        "grad_cam",
        "grad_cam_plusplus",
        "ablation_cam",
        "xgrad_cam",
        "layer_cam",
        "eigencam",
        "eigengrad_cam",
        "score_cam",
    ]:
        sys.modules[f"pytorch_grad_cam.{sub}"] = types.ModuleType(f"pytorch_grad_cam.{sub}")

# ---- Streamlit setup ----
st.set_page_config(page_title="Ovarian Cancer Prediction", page_icon="🧬", layout="centered")
st.title("🧬 Ovarian Cancer Detection App")
st.write("Upload a histopathology image to predict whether it's **Ovarian Cancer or Non-Ovarian Cancer**.")

# ---- Sidebar ----
st.sidebar.header("⚙️ Preprocessing & Calibration")
input_size = st.sidebar.selectbox("Input size (pixels)", [224, 256, 288, 384], index=0)
normalization = st.sidebar.selectbox("Normalization", ["ImageNet (default)", "0-1 (no normalize)"], index=0)
mean, std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) if normalization == "ImageNet (default)" else ([0,0,0], [1,1,1])
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
bias_factor = st.sidebar.slider("Bias factor (Ovarian Cancer multiplier)", 0.5, 2.0, 1.0, 0.01)
show_debug = st.sidebar.checkbox("Show debug info", False)
flip_labels = st.sidebar.checkbox("Reverse label interpretation", False)

# ---- Model Definition ----
class HybridCNNViT(nn.Module):
    def __init__(self, cnn_model_name="resnet50", vit_model_name="vit_base_patch16_224", num_classes=2):
        super().__init__()
        self.cnn = timm.create_model(cnn_model_name, pretrained=False, num_classes=0, global_pool="avg")
        self.vit = timm.create_model(vit_model_name, pretrained=False, num_classes=0, global_pool="avg")
        self.classifier = nn.Linear(self.cnn.num_features + self.vit.num_features, num_classes)
    def forward(self, x):
        return self.classifier(torch.cat((self.cnn(x), self.vit(x)), dim=1))

# 🔹 Force mapping so pickle knows where to find class
import builtins
builtins.HybridCNNViT = HybridCNNViT  # 👈 Key line to fix the "Can't get attribute" error

# ---- Model Download ----
@st.cache_resource
def ensure_model_downloaded(path="hybrid_models_fixed.pkl"):
    FILE_ID = "1fxzDo4lgzMORaQgwjcZe42Xq1F62IBQK"
    URL = f"https://drive.google.com/uc?id={FILE_ID}"
    if not os.path.exists(path):
        st.info("📥 Downloading model from Google Drive...")
        gdown.download(URL, path, quiet=False)
        st.success("✅ Model downloaded successfully.")
    return path

# ---- Load model ----
@st.cache_resource
def load_model():
    model_path = ensure_model_downloaded()
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        model.eval()
        st.success("✅ Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model()

# ---- Transform ----
def make_transform(size, mean, std):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
transform = make_transform(input_size, mean, std)

# ---- Prediction ----
def predict_tensor(model, tensor, threshold, bias_factor):
    model.eval()
    with torch.no_grad():
        logits = model(tensor).cpu().squeeze().numpy()
        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()
        probs[1] *= bias_factor
        probs /= probs.sum()
        pred = 1 if probs[1] >= threshold else 0
    return pred, probs, logits

# ---- UI ----
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    tensor = transform(img).unsqueeze(0)

    if st.button("🔍 Predict"):
        if model is None:
            st.warning("⚠️ Model not loaded.")
        else:
            pred, probs, logits = predict_tensor(model, tensor, threshold, bias_factor)
            labels = ["Non Ovarian Cancer", "Ovarian Cancer"]
            if flip_labels: labels = labels[::-1]
            st.subheader(f"🎯 Prediction: {labels[pred]}")
            st.write(f"{labels[0]}: {probs[0]*100:.2f}% | {labels[1]}: {probs[1]*100:.2f}%")
            st.progress(int(probs[1]*100))
            if show_debug:
                st.write("Logits:", logits)
                st.write("Probs:", probs)
