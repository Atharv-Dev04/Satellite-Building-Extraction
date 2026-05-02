import os
import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# ==============================
# ⚙️ CONFIGURATION
# ==============================
PATCH_DIR = r"D:\Research Project\Dataset\Image 5\images"
MODEL_ARCH = "linknet" # Options: unet, unetplusplus, manet, deeplabv3plus, linknet
WEIGHTS_PATH = r"D:\Research Project\results_v2\linknet\best_linknet_v2_resnet34.pth"

IMG_SIZE = 256
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output subfolder for the specific model (AUTO-GENERATED)
OUTPUT_DIR = os.path.join(r"D:\Research Project\Inference_Results_Image5", MODEL_ARCH)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# 🏗️ LOAD MODEL
# ==============================
def load_model():
    archs = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "manet": smp.MAnet,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "linknet": smp.Linknet
    }
    
    model = archs[MODEL_ARCH](
        encoder_name="resnet34",
        in_channels=3,
        classes=1
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    print(f"✅ Loaded {MODEL_ARCH} from {WEIGHTS_PATH}")
    return model

# ==============================
# 🔄 PREPROCESSING
# ==============================
transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def preprocess(img_path):
    import rasterio
    with rasterio.open(img_path) as src:
        image = src.read()
        image = np.transpose(image, (1, 2, 0)) # To HWC
    
    # Scaling optimization (16-bit to 8-bit using factor 500)
    orig_8bit = (image.astype('float32') / 500.0 * 255).astype('uint8')
    orig_8bit = np.clip(orig_8bit, 0, 255)
    
    # Resize and Transform
    image_resized = cv2.resize(orig_8bit, (IMG_SIZE, IMG_SIZE))
    augmented = transform(image=image_resized)
    return augmented['image'].unsqueeze(0).to(DEVICE), image_resized

# ==============================
# 🚀 MAIN INFERENCE LOOP
# ==============================
model = load_model()
patch_files = [f for f in os.listdir(PATCH_DIR) if f.endswith(".tif")]
results = []

print(f"🔍 Starting inference on {len(patch_files)} patches...")

for filename in tqdm(patch_files):
    path = os.path.join(PATCH_DIR, filename)
    
    # Predict
    input_tensor, display_img = preprocess(path)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).cpu().numpy()[0, 0]
        mask = (prob > THRESHOLD).astype(np.uint8)
    
    # Calculate coverage
    area_percent = (np.sum(mask) / mask.size) * 100
    results.append({"File": filename, "BuiltUp_Area_%": round(area_percent, 2)})
    
    # Save Visualization (every 5th patch to save space, or all if you prefer)
    if area_percent > 1.0: # Only save if we found something
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(display_img)
        ax[0].set_title("Original")
        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title(f"Prediction ({area_percent:.1f}%)")
        plt.savefig(os.path.join(OUTPUT_DIR, f"pred_{filename}.png"))
        plt.close()

# Save CSV
df = pd.DataFrame(results)
df.to_csv(os.path.join(OUTPUT_DIR, "inference_summary.csv"), index=False)
print(f"\n🎉 Done! Results saved in: {OUTPUT_DIR}")
