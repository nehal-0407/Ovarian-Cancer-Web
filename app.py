# app.py
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
import sys, types, builtins

# ------------------------------------------------
# 🧩 Patch: Mock pytorch_grad_cam so pickle can load safely
# ------------------------------------------------
if "pytorch_grad_cam" not in sys.modules:
    pkg = types.ModuleType("pytorch_grad_cam")
    pkg.__path__ = []
    sys.modules["pytorch_grad_cam"] = pkg

# create dummy submodules
submodules = [
    "activations_and_gradients", "base_cam", "grad_cam", "grad_cam_plusplus",
    "ablation_cam", "xgrad_cam", "layer_cam", "eigencam",
    "eigengrad_cam", "score_cam", "utils"
]
for sub in submodules:
    sys.modules[f"pytorch_grad_cam.{sub}"] = types.ModuleType(f"pytorch_grad_cam.{sub}")

# ✅ Robust dummy class with all missing methods
class ActivationsAndGradients:
    def __init__(self, *args, **kwargs):
        self.activations = []
        self.gradients = []
    def save_activation(self, *a, **kw): pass
    def save_gradient(self, *a, **kw): pass
    def release(self): pass
    def __call__(self, x): return x
    def __getstate__(self): return {}
    def __setstate__(self, state): pass

sys.modules["pytorch_grad_cam.activations_and_gradients"].ActivationsAndGradients = ActivationsAndGradients

# ------------------------------------------------
# 🌐 Streamlit setup
# ------------------------------------------------
st.set_page_config(page_title="Ovarian Cancer Prediction", page_icon="🧬", layout="centered")
st.title("🧬 Ovarian Cancer Detection App")
st.write("Upload a histopathology image to predict whether it's **Ovarian Cancer or Non-Ovarian Cancer**.")

# ------------------------------------------------
# ⚙️ Sidebar
# ------------------------------------------------
st.sidebar.header("⚙️ Preprocessing & Calibration")
input_size = st.sidebar.selectbox("Input size (pixels)", [224, 256, 288, 384], index=0)
normalization = st.sidebar.selectbox("Normalization", ["ImageNet (default)", "0-1 (no normalize)", "Custom"], index=0)

if normalization == "Custom":
    mean_str = st.sidebar.text_input("Mean (comma)", "0.485,0.456,0.406")
    std_str = st.sidebar.text_input("Std (comma)", "0.229,0.224,0.225")
    try:
        mean = [float(x.strip()) for x in mean_str.split(",")]
        std = [float(x.strip()) for x in std_str.split(",")]
    except:
        st.sidebar.error("⚠️ Error parsing custom mean/std — using defaults.")
        mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
elif normalization == "ImageNet (default)":
    mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
else:
    mean, std = [0.0,0.0,0.0], [1.0,1.0,1.0]

threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
temperature = st.sidebar.slider("Temperature", 0.5, 3.0, 1.5, 0.1)
bias_factor = st.sidebar.slider("Bias factor", 0.2, 2.0, 1.0, 0.01)
flip_labels = st.sidebar.checkbox("Reverse labels", False)
show_debug = st.sidebar.checkbox("Show raw logits", False)

# ------------------------------------------------
# 🧠 Model class
# ------------------------------------------------
class HybridCNNViT(nn.Module):
    def __init__(self, cnn_model_name="resnet50", vit_model_name="vit_base_patch16_224", num_classes=2):
        super(HybridCNNViT, self).__init__()
        self.cnn = timm.create_model(cnn_model_name, pretrained=False, num_classes=0, global_pool="avg")
        self.vit = timm.create_model(vit_model_name, pretrained=False, num_classes=0, global_pool="avg")
        combined_features = self.cnn.num_features + self.vit.num_features
        self.classifier = nn.Linear(combined_features, num_classes)
    def forward(self, x):
        return self.classifier(torch.cat((self.cnn(x), self.vit(x)), dim=1))

builtins.HybridCNNViT = HybridCNNViT  # so pickle can find it

# ------------------------------------------------
# 📥 Model load (patched for timm Attention.norm)
# ------------------------------------------------
@st.cache_resource
def load_model():
    path = ensure_model_downloaded()

    # --- Patch timm Attention if needed ---
    import timm.models.vision_transformer as vit_mod

    if hasattr(vit_mod, "Attention"):
        Attn = vit_mod.Attention
        # Add dummy .norm if it doesn't exist
        if not hasattr(Attn, "norm"):
            def fake_norm(self, x):
                return x
            Attn.norm = fake_norm
            # also handle missing norm attributes on instances
            old_init = Attn.__init__
            def new_init(self, *a, **kw):
                old_init(self, *a, **kw)
                self.norm = lambda x: x
            Attn.__init__ = new_init

    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        model.eval()
        st.success("✅ Model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# ------------------------------------------------
# 🧩 Preprocessing + Predict
# ------------------------------------------------
def make_transform(size, mean, std):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
transform = make_transform(input_size, mean, std)

def predict_tensor(model, input_tensor, threshold=0.5, temperature=1.0, bias_factor=1.0):
    model.eval()
    with torch.no_grad():
        out = model(input_tensor).squeeze(0).cpu().numpy() / temperature
        exp = np.exp(out - np.max(out))
        probs = exp / exp.sum()
        probs[1] *= bias_factor
        probs /= probs.sum()
        pred = 1 if probs[1] >= threshold else 0
        return pred, probs, out

# ------------------------------------------------
# 🖼️ UI
# ------------------------------------------------
uploaded = st.file_uploader("📤 Upload image", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    tensor = transform(img).unsqueeze(0)
    if st.button("🔍 Predict"):
        if model is None:
            st.warning("⚠️ Model not loaded.")
        else:
            pred, probs, raw = predict_tensor(model, tensor, threshold, temperature, bias_factor)
            labels = ["Non Ovarian Cancer", "Ovarian Cancer"]
            if flip_labels: labels.reverse()
            st.subheader(f"🎯 Prediction: {labels[pred]}")
            st.write(f"{labels[0]}: {probs[0]*100:.2f}% | {labels[1]}: {probs[1]*100:.2f}%")
            st.progress(int(probs[1]*100))
            if show_debug:
                st.write("Raw logits:", raw)
