# final_app.py
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


# --- Patch for missing pytorch_grad_cam on Streamlit Cloud ---
if "pytorch_grad_cam" not in sys.modules:
    fake_cam = types.ModuleType("pytorch_grad_cam")
    sys.modules["pytorch_grad_cam"] = fake_cam
    # Fake required submodules so unpickling works
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
        submodule = types.ModuleType(f"pytorch_grad_cam.{sub}")
        setattr(fake_cam, sub, submodule)
        sys.modules[f"pytorch_grad_cam.{sub}"] = submodule


# ----------------------------
# Streamlit setup
# ----------------------------
st.set_page_config(page_title="Ovarian Cancer Prediction", page_icon="🧬", layout="centered")
st.title("🧬 Ovarian Cancer Detection App")
st.write("Upload a histopathology image to predict whether it's **Ovarian Cancer or Non-Ovarian Cancer**.")


# ----------------------------
# Sidebar configuration
# ----------------------------
st.sidebar.header("⚙️ Preprocessing & Calibration")
input_size = st.sidebar.selectbox("Input size (pixels)", [224, 256, 288, 384], index=0)

normalization = st.sidebar.selectbox(
    "Normalization", ["ImageNet (default)", "0-1 (no normalize)", "Custom"], index=0
)

if normalization == "Custom":
    mean_str = st.sidebar.text_input("Mean (comma)", "0.485,0.456,0.406")
    std_str = st.sidebar.text_input("Std (comma)", "0.229,0.224,0.225")
    try:
        mean = [float(x.strip()) for x in mean_str.split(",")]
        std = [float(x.strip()) for x in std_str.split(",")]
    except:
        st.sidebar.error("⚠️ Error parsing custom mean/std — using defaults.")
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
elif normalization == "ImageNet (default)":
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
else:
    mean = [0.0, 0.0, 0.0]
    std = [1.0, 1.0, 1.0]

threshold = st.sidebar.slider("Decision threshold for Cancer", 0.0, 1.0, 0.50, 0.01)
temperature = st.sidebar.slider("Temperature (confidence calibration)", 0.5, 3.0, 1.5, 0.1)
bias_factor = st.sidebar.slider("Bias factor (Ovarian Cancer multiplier)", 0.5, 2.0, 1.0, 0.01)

flip_labels = st.sidebar.checkbox("Reverse label interpretation", False)
show_debug = st.sidebar.checkbox("Show raw logits/probabilities", False)

st.sidebar.markdown("---")
st.sidebar.caption("If your model over-predicts cancer, try reducing Bias factor (<1).")


# ----------------------------
# Hybrid model definition
# ----------------------------
class HybridCNNViT(nn.Module):
    def __init__(self, cnn_model_name="resnet50", vit_model_name="vit_base_patch16_224", num_classes=2):
        super(HybridCNNViT, self).__init__()
        self.cnn = timm.create_model(cnn_model_name, pretrained=False, num_classes=0, global_pool="avg")
        self.vit = timm.create_model(vit_model_name, pretrained=False, num_classes=0, global_pool="avg")
        combined_features = self.cnn.num_features + self.vit.num_features
        self.classifier = nn.Linear(combined_features, num_classes)

    def forward(self, x):
        cnn_features = self.cnn(x)
        vit_features = self.vit(x)
        combined = torch.cat((cnn_features, vit_features), dim=1)
        return self.classifier(combined)


# ----------------------------
# Download model from Google Drive
# ----------------------------
@st.cache_resource
def ensure_model_downloaded(local_path="hybrid_models_fixed.pkl"):
    FILE_ID = "1fxzDo4lgzMORaQgwjcZe42Xq1F62IBQK"
    DRIVE_URL = f"https://drive.google.com/uc?id={FILE_ID}"

    if not os.path.exists(local_path):
        st.info("📥 Downloading model from Google Drive...")
        try:
            gdown.download(DRIVE_URL, local_path, quiet=False)
            st.success("✅ Model downloaded successfully.")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
    return local_path


# ----------------------------
# Load model safely
# ----------------------------
@st.cache_resource
def load_model():
    model_path = ensure_model_downloaded()
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Add classifier if missing
        if not hasattr(model, "classifier"):
            cnn_f = getattr(model.cnn, "num_features", 2048)
            vit_f = getattr(model.vit, "num_features", 768)
            model.classifier = nn.Linear(cnn_f + vit_f, 2)
            st.info("🔧 Classifier added (auto-fixed).")

        model.eval()
        return model

    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None


model = load_model()


# ----------------------------
# Preprocessing
# ----------------------------
def make_transform(size, mean, std):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


transform = make_transform(input_size, mean, std)


# ----------------------------
# Prediction helper
# ----------------------------
def predict_tensor(model, input_tensor, threshold=0.5, temperature=1.0, bias_factor=1.0):
    device = torch.device("cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        out = outputs.detach().cpu().squeeze(0)
        if out.ndim == 0:
            out = out.unsqueeze(0)
        logits = out.numpy() / float(temperature)

        exp = np.exp(logits - np.max(logits))
        probs = exp / exp.sum()

        probs[1] = float(probs[1]) * float(bias_factor)
        probs = probs / probs.sum()

        pred = 1 if probs[1] >= threshold else 0
    return pred, probs, logits


# ----------------------------
# UI: Upload + Predict
# ----------------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    input_tensor = transform(image).unsqueeze(0)

    if st.button("🔍 Predict"):
        if model is None:
            st.warning("⚠️ Model not loaded.")
        else:
            pred, probs, raw_logits = predict_tensor(model, input_tensor, threshold, temperature, bias_factor)
            labels = ["Non Ovarian Cancer", "Ovarian Cancer"]
            if flip_labels:
                labels = labels[::-1]
            pred_label = labels[pred]

            st.subheader(f"🎯 Prediction: {pred_label}")
            st.write(f"Probabilities — {labels[0]}: {probs[0]*100:.2f}%, {labels[1]}: {probs[1]*100:.2f}%")
            st.progress(int(probs[1]*100))
            if show_debug:
                st.write("Raw logits:", raw_logits)
                st.write("Post-bias probabilities:", probs)

            diff = abs(probs[1] - probs[0])
            if probs[1] > 0.90:
                st.info("🧠 Model is highly confident — strong Ovarian Cancer signal.")
            elif probs[0] > 0.90:
                st.info("🩺 Model is highly confident — strong Non-Ovarian Cancer signal.")
            elif diff < 0.12:
                st.warning("⚠️ Model uncertain — probabilities are close. Try adjusting parameters.")
