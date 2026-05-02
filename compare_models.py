import os
import torch
import numpy as np
import rasterio
import cv2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ======================
# CONFIG
# ======================
IMG_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCALING_FACTOR = 500.0

# Image and Mask paths (Selected candidate)
IMAGE_PATH = "dataset_split/train/images/img1_patch_11.tif"
MASK_PATH = "dataset_split/train/masks/img1_patch_11.tif"

# Model information
MODEL_INFO = [
    {"name": "UNet", "arch": smp.Unet, "weights": "results_v2/unet/best_unet_v2_resnet34.pth"},
    {"name": "UNet++", "arch": smp.UnetPlusPlus, "weights": "results_v2/unetplusplus/best_unetplusplus_v2_resnet34.pth"},
    {"name": "MAnet", "arch": smp.MAnet, "weights": "results_v2/manet/best_manet_v2_resnet34.pth"},
    {"name": "DeepLabV3+", "arch": smp.DeepLabV3Plus, "weights": "results_v2/deeplabv3plus/best_deeplabv3plus_v2_resnet34.pth"},
    {"name": "LinkNet", "arch": smp.Linknet, "weights": "results_v2/linknet/best_linknet_v2_resnet34.pth"},
]

# ======================
# PREPROCESSING
# ======================
transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def preprocess_image(img_path):
    with rasterio.open(img_path) as src:
        image = src.read()
        image = np.transpose(image, (1, 2, 0))
    
    # Scaling optimization (16-bit to 8-bit using factor 500)
    if image.dtype != np.uint8:
        image = (image.astype('float32') / SCALING_FACTOR * 255).astype('uint8')
        image = np.clip(image, 0, 255)
    
    # Resize to model input size
    image_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Apply normalization
    augmented = transform(image=image_resized)
    image_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    
    return image_tensor, image_resized

def preprocess_mask(mask_path):
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    return (mask_resized > 0).astype('float32')

# ======================
# INFERENCE
# ======================
def get_predictions():
    print(f"Loading image: {IMAGE_PATH}")
    image_tensor, image_display = preprocess_image(IMAGE_PATH)
    mask_gt = preprocess_mask(MASK_PATH)
    
    results = {
        "Original Image": image_display,
        "Ground Truth": mask_gt
    }
    
    for info in MODEL_INFO:
        print(f"Running inference for {info['name']}...")
        # Initialize model
        model = info['arch'](
            encoder_name="resnet34",
            encoder_weights=None, # Loading local weights
            in_channels=3,
            classes=1
        ).to(DEVICE)
        
        # Load weights
        if os.path.exists(info['weights']):
            model.load_state_dict(torch.load(info['weights'], map_location=DEVICE))
            model.eval()
            
            with torch.no_grad():
                output = model(image_tensor)
                pred = torch.sigmoid(output).cpu().numpy()[0, 0]
                results[info['name']] = (pred > 0.5).astype(np.uint8)
        else:
            print(f"WARNING: Weights not found for {info['name']} at {info['weights']}")
            results[info['name']] = np.zeros((IMG_SIZE, IMG_SIZE))
            
    return results

# ======================
# VISUALIZATION
# ======================
def visualize(results):
    print("Generating updated visualization (Row-per-model layout)...")
    
    orig = results["Original Image"]
    gt = results["Ground Truth"]
    
    # We want 5 rows (one for each model) and 3 columns (Original, Truth, Pred)
    fig, axes = plt.subplots(5, 3, figsize=(15, 20))
    fig.suptitle(f"Model Comparison | Coverage: ~59.1%\nImage: {os.path.basename(IMAGE_PATH)}", fontsize=20)
    
    model_names = [info["name"] for info in MODEL_INFO]
    
    for i, model_name in enumerate(model_names):
        pred = results[model_name]
        
        # Original Image
        axes[i, 0].imshow(orig)
        if i == 0: axes[i, 0].set_title("Original Image", fontsize=16)
        axes[i, 0].set_ylabel(model_name, fontsize=16, fontweight='bold')
        
        # Ground Truth
        axes[i, 1].imshow(gt, cmap='gray')
        if i == 0: axes[i, 1].set_title("Ground Truth", fontsize=16)
        
        # Prediction
        axes[i, 2].imshow(pred, cmap='gray')
        if i == 0: axes[i, 2].set_title("Model Prediction", fontsize=16)
        
        # Turn off ticks for all
        for j in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("prediction_visualize.png", dpi=300, bbox_inches='tight')
    print("Saved updated visualization to prediction_visualize.png")

if __name__ == "__main__":
    preds = get_predictions()
    visualize(preds)
