import numpy as np
import torch
import rasterio
import io

from flask import Flask, request, render_template, send_file
import base64
from PIL import Image
from model import load_model

# -------------------------------
# App Initialization
# -------------------------------

app = Flask(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model + normalization stats once at startup
model, band_mins, band_maxs = load_model()
model.eval()


# -------------------------------
# Normalization Function
# -------------------------------

def normalize_per_band(img, band_mins, band_maxs):
    img_norm = np.zeros_like(img, dtype=np.float32)

    for b in range(img.shape[-1]):
        img_norm[..., b] = (
            (img[..., b] - band_mins[b]) /
            (band_maxs[b] - band_mins[b] + 1e-8)
        )

    return img_norm


# -------------------------------
# Routes
# -------------------------------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files["file"]

    # Read TIFF
    with rasterio.open(file) as src:
        img = src.read()  # (bands, H, W)
        img = np.transpose(img, (1, 2, 0)).astype(np.float32)

        # --------------------------
        # COMPUTE WATER INDICES
        # --------------------------

        GREEN = img[:, :, 2]
        NIR   = img[:, :, 4]
        SWIR1 = img[:, :, 5]
        SWIR2 = img[:, :, 6]

        ndwi = (GREEN - NIR) / (GREEN + NIR + 1e-8)
        mndwi = (GREEN - SWIR1) / (GREEN + SWIR1 + 1e-8)
        awei = 4*(GREEN - SWIR1) - (0.25*NIR + 2.75*SWIR2)

        indices = np.stack([ndwi, mndwi, awei], axis=-1)

        # Add indices to image
        img = np.concatenate([img, indices], axis=-1)

    if img.shape[0] != 128 or img.shape[1] != 128:
        return {"error": "Invalid image size (Expected 128x128)"}, 400

    # --------------------------
    # ORIGINAL RGB CONVERSION
    # --------------------------
    rgb = img[:, :, :3]

    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    rgb = (rgb * 255).astype(np.uint8)

    rgb_pil = Image.fromarray(rgb)
    rgb_buffer = io.BytesIO()
    rgb_pil.save(rgb_buffer, format="PNG")
    rgb_base64 = base64.b64encode(rgb_buffer.getvalue()).decode("utf-8")

    # --------------------------
    # NORMALIZE FOR MODEL
    # --------------------------
    img_norm = normalize_per_band(img, band_mins, band_maxs)

    img_tensor = torch.tensor(img_norm) \
        .permute(2, 0, 1) \
        .unsqueeze(0) \
        .float() \
        .to(DEVICE)

    # --------------------------
    # INFERENCE
    # --------------------------
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.sigmoid(output)
        pred = (pred > 0.5).float()

    mask = pred.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)

    mask_pil = Image.fromarray(mask)
    mask_buffer = io.BytesIO()
    mask_pil.save(mask_buffer, format="PNG")
    mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode("utf-8")

    return {
        "original": rgb_base64,
        "mask": mask_base64
    }


# -------------------------------
# Run App
# -------------------------------

if __name__ == "__main__":
    app.run(debug=True)
    # app.run(host="0.0.0.0", port=5000)