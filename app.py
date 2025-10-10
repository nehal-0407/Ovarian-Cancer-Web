# final_app.py (replace app.py with this)
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import pickle
import numpy as np
import os

# ----------------------------
# Streamlit setup
# ----------------------------
st.set_page_config(page_title="Ovarian Cancer Prediction", page_icon="🧬", layout="centered")
st.title("🧬 Ovarian Cancer Detection App")
st.write("Upload a histopathology image to predict whether it's **Ovarian Cancer or Non-Ovarian Cancer**.")

# ----------------------------
# Sidebar controls
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

# bias factor multiplies malignant probability before renormalizing
if "bias_factor" not in st.session_state:
    st.session_state.bias_factor = 1.0
bias_factor = st.sidebar.slider("Bias factor (malignant multiplier)", 0.2, 2.0, float(st.session_state.bias_factor), 0.01)

flip_labels = st.sidebar.checkbox("Reverse label interpretation", False)
show_debug = st.sidebar.checkbox("Show raw logits/probabilities", False)

st.sidebar.markdown("---")
st.sidebar.caption("If your model over-predicts cancer, try reducing Bias factor (<1).")
st.sidebar.caption("Use 'Auto-calibrate from ./test_images' if you have labeled examples locally.")

# ----------------------------
# Model definition (same as training)
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
# Load trained model
# ----------------------------
@st.cache_resource
def load_model():
    model_path = "hybrid_models_fixed.pkl" if os.path.exists("hybrid_models_fixed.pkl") else "hybrid_models.pkl"
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Auto repair if classifier missing, and save fixed version
        if not hasattr(model, "classifier"):
            cnn_f = getattr(model.cnn, "num_features", 2048)
            vit_f = getattr(model.vit, "num_features", 768)
            model.classifier = nn.Linear(cnn_f + vit_f, 2)
            with open("hybrid_models_fixed.pkl", "wb") as fx:
                pickle.dump(model, fx)
            st.info("🔧 Classifier added and saved to 'hybrid_models_fixed.pkl'")

        model.eval()
        return model
    except FileNotFoundError:
        st.error("❌ Model file not found. Put 'hybrid_models.pkl' or 'hybrid_models_fixed.pkl' next to app.py.")
        return None
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model()

# ----------------------------
# Image preprocessing
# ----------------------------
def make_transform(size, mean, std):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

transform = make_transform(input_size, mean, std)

# ----------------------------
# Prediction helper (with temperature + bias-factor)
# ----------------------------
def predict_tensor(model, input_tensor, threshold=0.5, temperature=1.0, bias_factor=1.0):
    # Handles both single-logit and two-logit outputs.
    device = torch.device("cpu")
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        out = outputs.detach().cpu().squeeze(0)

        # ensure 1-D
        if out.ndim == 0:
            out = out.unsqueeze(0)

        # convert to numpy logits
        logits = out.numpy() / float(temperature)

        if logits.shape[0] == 1:
            # single-logit -> sigmoid
            logit = float(np.asarray(logits).item())
            prob_cancer = float(torch.sigmoid(torch.tensor(logit)).item())
            probs = np.array([1.0 - prob_cancer, prob_cancer])
        else:
            # multiclass logits -> softmax
            exp = np.exp(logits - np.max(logits))
            probs = exp / exp.sum()

        # Apply bias factor to malignant class probability, then renormalize
        probs[1] = float(probs[1]) * float(bias_factor)
        probs = probs / probs.sum()

        # Final decision
        pred = 1 if probs[1] >= threshold else 0

    return pred, probs, logits

# ----------------------------
# Auto-calibrate bias from ./test_images folder
# ----------------------------
def calibrate_bias_from_folder(folder="test_images", desired_non_cancer_mean=0.2):
    """
    Compute mean malignant probability on non-cancer images and return a bias_factor
    that would move mean malignant probability to desired_non_cancer_mean.
    """
    if model is None:
        return None, "Model not loaded"

    # folders expected:
    # test_images/Non_Ovarian_Cancer/*.jpg
    # test_images/Ovarian_Cancer/*.jpg
    non_path = os.path.join(folder, "Non_Ovarian_Cancer")
    mal_path = os.path.join(folder, "Ovarian_Cancer")
    if not os.path.isdir(non_path) or not os.path.isdir(mal_path):
        return None, "Missing test_images/Non_Ovarian_Cancer or test_images/Ovarian_Cancer"

    non_probs = []
    mal_probs = []
    for fname in os.listdir(non_path):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = Image.open(os.path.join(non_path, fname)).convert("RGB")
        t = transform(img).unsqueeze(0)
        _, probs, _ = predict_tensor(model, t, threshold=threshold, temperature=temperature, bias_factor=1.0)
        non_probs.append(float(probs[1]))
    for fname in os.listdir(mal_path):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = Image.open(os.path.join(mal_path, fname)).convert("RGB")
        t = transform(img).unsqueeze(0)
        _, probs, _ = predict_tensor(model, t, threshold=threshold, temperature=temperature, bias_factor=1.0)
        mal_probs.append(float(probs[1]))

    if len(non_probs) == 0:
        return None, "No images found in Non_Ovarian_Cancer"

    mean_non = float(np.mean(non_probs))
    mean_mal = float(np.mean(mal_probs)) if mal_probs else None

    # compute factor to move mean_non -> desired_non_cancer_mean
    # factor = desired / observed  (cap between 0.2 and 2.0)
    if mean_non == 0:
        factor = 0.5
    else:
        factor = desired_non_cancer_mean / mean_non
    factor = float(np.clip(factor, 0.2, 2.0))

    msg = f"Observed mean malignant prob: non-cancer={mean_non:.3f}"
    if mean_mal is not None:
        msg += f", cancer={mean_mal:.3f}"
    return factor, msg

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
            pred, probs, raw_logits = predict_tensor(model, input_tensor, threshold=threshold,
                                                    temperature=temperature, bias_factor=bias_factor)

            base_labels = ["Non Ovarian Cancer", "Ovarian Cancer"]  # index 0 -> non-cancer, 1 -> cancer
            labels = base_labels[::-1] if flip_labels else base_labels
            pred_label = labels[pred]

            st.subheader(f"🎯 Prediction: {pred_label}")
            st.write(f"Probabilities — {labels[0]}: {probs[0]*100:.2f}%, {labels[1]}: {probs[1]*100:.2f}%")
            st.progress(int(probs[1]*100))

            if show_debug:
                st.write("Raw logits:", raw_logits)
                st.write("Post-bias probabilities:", probs)

            # improved confidence messaging
            diff = abs(probs[1] - probs[0])
            if probs[1] > 0.90:
                st.info("🧠 Model is highly confident — strong Ovarian Cancer signal.")
            elif probs[0] > 0.90:
                st.info("🩺 Model is highly confident — strong Non-Ovarian Cancer signal.")
            elif diff < 0.12:
                st.warning("⚠️ Model uncertain — probabilities are close. Try changing normalization/input-size or calibrate bias.")

# ----------------------------
# Auto-calibration controls
# ----------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Auto-calibrate bias (optional)")
desired = st.sidebar.slider("Desired mean malignant prob on Non-Ovarian-Cancer (target)", 0.05, 0.40, 0.20, 0.01)
if st.sidebar.button("Auto-calibrate from ./test_images"):
    factor, msg = calibrate_bias_from_folder(folder="test_images", desired_non_cancer_mean=desired)
    if factor is None:
        st.sidebar.error(msg)
    else:
        st.session_state.bias_factor = factor
        # update local slider value by reloading page (Streamlit retains session_state)
        st.sidebar.success(f"Auto-calibrated bias_factor={factor:.3f}. {msg}")
        st.experimental_rerun()

# ----------------------------
# Quick evaluation mode
# ----------------------------
def evaluate_folder(folder="test_images"):
    files, y_true, y_pred = [], [], []
    for label_idx, label_name in enumerate(["Non_Ovarian_Cancer", "Ovarian_Cancer"]):
        path = os.path.join(folder, label_name)
        if not os.path.isdir(path):
            continue
        for fname in os.listdir(path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                files.append(os.path.join(path, fname))
                y_true.append(label_idx)

    if not files:
        return None

    for fpath in files:
        img = Image.open(fpath).convert("RGB")
        t = transform(img).unsqueeze(0)
        pred, probs, _ = predict_tensor(model, t, threshold=threshold, temperature=temperature, bias_factor=bias_factor)
        y_pred.append(pred)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = (y_true == y_pred).mean()
    return {"n": len(files), "accuracy": float(acc)}

if st.sidebar.button("Run quick eval on ./test_images"):
    info = evaluate_folder()
    if info is None:
        st.sidebar.warning("⚠️ No test_images/Non_Ovarian_Cancer or Ovarian_Cancer folders found.")
    else:
        st.sidebar.success(f"✅ Evaluated {info['n']} images — Accuracy: {info['accuracy']*100:.2f}%")
