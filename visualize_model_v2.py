import os
import numpy as np
import torch
import segmentation_models_pytorch as smp
import rasterio
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# ======================
# CONFIG (V2 OPTIMIZED)
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256

# --- CHOOSE ARCHITECTURE HERE ---
# Options: "unet", "unetplusplus", "linknet", "deeplabv3plus", "manet"
ARCH_CHOICE = "unet" 

MODEL_NAME = f"{ARCH_CHOICE}_v2_resnet34"
OUTPUT_DIR = os.path.join("results_v2", ARCH_CHOICE)
MODEL_PATH = os.path.join(OUTPUT_DIR, f"best_{MODEL_NAME}.pth")
TEST_IMG_DIR = "dataset_split/test/images"
TEST_MASK_DIR = "dataset_split/test/masks"

# ======================
# DATASET
# ======================
class SatelliteTestDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.samples = [f for f in os.listdir(img_dir) if f.endswith(".tif")]
        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
        print(f"Loaded {len(self.samples)} test samples")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        filename = self.samples[idx]
        img_path = os.path.join(self.img_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)
        with rasterio.open(img_path) as src:
            image = src.read(); image = np.transpose(image, (1, 2, 0))
        with rasterio.open(mask_path) as src: mask = src.read(1)
        
        # V2 Scaling Optimization: Factor 500
        if image.dtype == np.uint16 or image.dtype != np.uint8:
            image_8bit = (image.astype('float32') / 500.0 * 255).astype('uint8')
            image_8bit = np.clip(image_8bit, 0, 255)
        else: image_8bit = image

        image_resized = cv2.resize(image_8bit, (IMG_SIZE, IMG_SIZE))
        mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask_binary = (mask_resized > 0).astype('float32')
        
        augmented = self.transform(image=image_resized, mask=mask_binary)
        return augmented['image'], torch.from_numpy(mask_binary), image_resized

def get_model(arch):
    if arch == "unet": return smp.Unet
    if arch == "unetplusplus": return smp.UnetPlusPlus
    if arch == "linknet": return smp.Linknet
    if arch == "deeplabv3plus": return smp.DeepLabV3Plus
    if arch == "manet": return smp.MAnet
    raise ValueError(f"Unknown architecture: {arch}")

def visualize_v2():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model weights not found at {MODEL_PATH}"); return

    print(f"Loading V2 Model: {MODEL_NAME}...")
    model_class = get_model(ARCH_CHOICE)
    model = model_class(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    dataset = SatelliteTestDataset(TEST_IMG_DIR, TEST_MASK_DIR)
    loader = DataLoader(dataset, batch_size=5, shuffle=True)
    images, masks, raw_images = next(iter(loader))

    with torch.no_grad():
        preds = torch.sigmoid(model(images.to(DEVICE))).cpu().numpy()

    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    fig.suptitle(f"V2 Visual Evaluation: {ARCH_CHOICE.upper()}", fontsize=20)
    
    for i in range(min(5, len(images))):
        axes[i, 0].imshow(raw_images[i]); axes[i, 0].set_title("Original (V2 Scaled)")
        axes[i, 1].imshow(masks[i], cmap='gray'); axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(preds[i, 0] > 0.5, cmap='gray'); axes[i, 2].set_title(f"Prediction")
        for j in range(3): axes[i, j].axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    output_png = os.path.join(OUTPUT_DIR, f"predictions_v2_{ARCH_CHOICE}.png")
    plt.savefig(output_png); print(f"Success! Visualization saved to {output_png}")

if __name__ == "__main__":
    visualize_v2()
