import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import rasterio
from torch.utils.data import Dataset, DataLoader

# ======================
# CONFIG (V1 BASELINE)
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

# --- CHOOSE ARCHITECTURE HERE ---
# Options: "unet", "unetplusplus", "linknet", "deeplabv3plus", "manet"
ARCH_CHOICE = "unet" 

MODEL_NAME = f"{ARCH_CHOICE}_v1_resnet34"
OUTPUT_DIR = os.path.join("results", ARCH_CHOICE)
MODEL_PATH = os.path.join(OUTPUT_DIR, f"best_{MODEL_NAME}.pth")
TEST_IMG_DIR = "dataset_split/test/images"
TEST_MASK_DIR = "dataset_split/test/masks"

# ======================
# DATASET
# ======================
class SatelliteDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.samples = []
        if not os.path.exists(img_dir): return
        file_list = [f for f in os.listdir(img_dir) if f.endswith(".tif")]
        for filename in file_list:
            img_path = os.path.join(img_dir, filename); mask_path = os.path.join(mask_dir, filename)
            if os.path.exists(mask_path): self.samples.append((img_path, mask_path))
        print(f"Loaded {len(self.samples)} test samples from {img_dir}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        with rasterio.open(img_path) as src:
            image = src.read(); image = np.transpose(image, (1, 2, 0))
        with rasterio.open(mask_path) as src: mask = src.read(1)

        # V1 Scaling: Factor 1000
        if image.dtype == np.uint16 or image.dtype != np.uint8:
            image = (image.astype('float32') / 1000.0 * 255).astype('uint8')
            image = np.clip(image, 0, 255)

        image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_resized > 0).astype('float32')

        # Normalization
        image_norm = image_resized / 255.0
        mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
        image_norm = (image_norm - mean) / std; image_norm = np.transpose(image_norm, (2, 0, 1))

        return torch.tensor(image_norm, dtype=torch.float32), \
               torch.tensor(mask_binary, dtype=torch.float32).unsqueeze(0), \
               image_resized

def get_model(arch):
    if arch == "unet": return smp.Unet
    if arch == "unetplusplus": return smp.UnetPlusPlus
    if arch == "linknet": return smp.Linknet
    if arch == "deeplabv3plus": return smp.DeepLabV3Plus
    if arch == "manet": return smp.MAnet
    raise ValueError(f"Unknown architecture: {arch}")

def visualize():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model weights not found at {MODEL_PATH}"); return

    print(f"Loading V1 Model: {MODEL_NAME}...")
    model_class = get_model(ARCH_CHOICE)
    model = model_class(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    test_dataset = SatelliteDataset(TEST_IMG_DIR, TEST_MASK_DIR)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True)
    images, masks, raw_images = next(iter(test_loader))

    with torch.no_grad():
        preds = torch.sigmoid(model(images.to(DEVICE))).cpu().numpy()

    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    fig.suptitle(f"V1 Visual Evaluation: {ARCH_CHOICE.upper()}", fontsize=20)
    
    for i in range(min(5, len(images))):
        axes[i, 0].imshow(raw_images[i]); axes[i, 0].set_title("Original (V1 Scaled)")
        axes[i, 1].imshow(masks[i].squeeze(), cmap='gray'); axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(preds[i, 0] > 0.5, cmap='gray'); axes[i, 2].set_title(f"Prediction")
        for j in range(3): axes[i, j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_png = os.path.join(OUTPUT_DIR, f"predictions_v1_{ARCH_CHOICE}.png")
    plt.savefig(output_png); print(f"Success! Visualization saved to {output_png}")

if __name__ == "__main__":
    visualize()
