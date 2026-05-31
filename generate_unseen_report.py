import os
import random
import numpy as np
import torch
import cv2
import rasterio
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set random seed for reproducibility (optional, but good for consistent premium results)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
SCALING_FACTOR = 500.0
TEST_IMG_DIR = "dataset_split/test/images"
TEST_MASK_DIR = "dataset_split/test/masks"
OUTPUT_PATH = "unseen_inference_report.png"

# Model definitions and weights
MODEL_INFO = [
    {"name": "UNet", "arch": smp.Unet, "weights": "results_v2/unet/best_unet_v2_resnet34.pth"},
    {"name": "UNet++", "arch": smp.UnetPlusPlus, "weights": "results_v2/unetplusplus/best_unetplusplus_v2_resnet34.pth"},
    {"name": "MAnet", "arch": smp.MAnet, "weights": "results_v2/manet/best_manet_v2_resnet34.pth"},
    {"name": "DeepLabV3+", "arch": smp.DeepLabV3Plus, "weights": "results_v2/deeplabv3plus/best_deeplabv3plus_v2_resnet34.pth"},
    {"name": "LinkNet", "arch": smp.Linknet, "weights": "results_v2/linknet/best_linknet_v2_resnet34.pth"},
]

# Image Preprocessing Transform
transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def preprocess_image(img_path):
    with rasterio.open(img_path) as src:
        image = src.read()
        image = np.transpose(image, (1, 2, 0))
    
    # 16-bit to 8-bit scaling using optimized factor 500
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

def calculate_metrics(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union > 0 else (1.0 if pred.sum() == 0 and gt.sum() == 0 else 0.0)
    
    dice = (2. * intersection) / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else (1.0 if pred.sum() == 0 and gt.sum() == 0 else 0.0)
    return iou, dice

def select_five_interesting_images():
    print("Scanning test set to select interesting patches...")
    all_files = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(".tif")]
    
    candidates = []
    for filename in all_files:
        mask_path = os.path.join(TEST_MASK_DIR, filename)
        if os.path.exists(mask_path):
            mask = preprocess_mask(mask_path)
            # Building pixel percentage
            pct = np.mean(mask) * 100.0
            # We want patches with 3% to 40% building pixels to ensure they are visually interesting
            if 3.0 <= pct <= 40.0:
                candidates.append((filename, pct))
    
    print(f"Found {len(candidates)} candidate patches with significant building presence.")
    
    # If we don't have enough candidates, relax the thresholds
    if len(candidates) < 5:
        candidates = []
        for filename in all_files:
            mask_path = os.path.join(TEST_MASK_DIR, filename)
            if os.path.exists(mask_path):
                mask = preprocess_mask(mask_path)
                pct = np.mean(mask) * 100.0
                if pct > 0.5: # just some buildings
                    candidates.append((filename, pct))
        print(f"Relaxed criteria: Found {len(candidates)} candidate patches.")
        
    if len(candidates) < 5:
        # Just take any files
        candidates = [(f, 0.0) for f in all_files[:5]]
        print("Fallback: selected first 5 files.")
        
    # Shuffle and select 5
    random.shuffle(candidates)
    selected = candidates[:5]
    print("Selected patches:")
    for name, pct in selected:
        print(f" - {name} (Building pixel coverage: {pct:.2f}%)")
        
    return [name for name, _ in selected]

def load_models():
    models = {}
    for info in MODEL_INFO:
        name = info["name"]
        weights_path = info["weights"]
        arch = info["arch"]
        
        print(f"Loading {name} model...")
        model = arch(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1
        ).to(DEVICE)
        
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
            model.eval()
            models[name] = model
            print(f"Loaded weights for {name} from {weights_path}")
        else:
            print(f"WARNING: Weights not found for {name} at {weights_path}")
            models[name] = None
            
    return models

def run_inference():
    selected_files = select_five_interesting_images()
    models = load_models()
    
    results = []
    
    for idx, filename in enumerate(selected_files):
        print(f"\nProcessing image {idx+1}/5: {filename}")
        img_path = os.path.join(TEST_IMG_DIR, filename)
        mask_path = os.path.join(TEST_MASK_DIR, filename)
        
        image_tensor, image_display = preprocess_image(img_path)
        mask_gt = preprocess_mask(mask_path)
        
        img_results = {
            "filename": filename,
            "image": image_display,
            "gt": mask_gt,
            "predictions": {}
        }
        
        for name, model in models.items():
            if model is not None:
                with torch.no_grad():
                    output = model(image_tensor)
                    pred = torch.sigmoid(output).cpu().numpy()[0, 0]
                    pred_binary = (pred > 0.5).astype(np.uint8)
                    
                    iou, dice = calculate_metrics(pred_binary, mask_gt)
                    img_results["predictions"][name] = {
                        "mask": pred_binary,
                        "iou": iou,
                        "dice": dice
                    }
            else:
                img_results["predictions"][name] = {
                    "mask": np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8),
                    "iou": 0.0,
                    "dice": 0.0
                }
                
        results.append(img_results)
        
    return results

def create_report_image(results):
    print("\nCreating black & white inference report image with default theme...")
    
    # 5 rows (one for each image), 7 columns:
    # Col 1: Original Image, Col 2: Ground Truth, Col 3-7: Models
    num_rows = 5
    num_cols = 7
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(22, 16))
    
    # Column headers
    headers = ["Original Image", "Ground Truth"] + [info["name"] for info in MODEL_INFO]
    
    for col_idx, text in enumerate(headers):
        axes[0, col_idx].annotate(text, xy=(0.5, 1.15), xycoords='axes fraction',
                                  ha='center', va='bottom', fontsize=18, fontweight='bold')
        
    for row_idx, res in enumerate(results):
        # Row label (Image filename or index)
        axes[row_idx, 0].annotate(f"Patch {row_idx + 1}\n{res['filename']}", xy=(-0.15, 0.5), xycoords='axes fraction',
                                  ha='right', va='center', fontsize=12, fontweight='bold', rotation=0)
        
        # 1. Original Image
        axes[row_idx, 0].imshow(res["image"])
        axes[row_idx, 0].axis('off')
        
        # 2. Ground Truth (Black & White)
        axes[row_idx, 1].imshow(res["gt"], cmap='gray')
        axes[row_idx, 1].axis('off')
        
        # Models
        for model_idx, model_info in enumerate(MODEL_INFO):
            name = model_info["name"]
            pred_data = res["predictions"][name]
            pred_mask = pred_data["mask"]
            iou = pred_data["iou"]
            dice = pred_data["dice"]
            
            ax = axes[row_idx, model_idx + 2]
            # Black & White prediction
            ax.imshow(pred_mask, cmap='gray')
            ax.axis('off')
            
            # Print performance metrics under the image
            metrics_text = f"IoU: {iou:.3f}\nDice: {dice:.3f}"
            ax.text(0.5, -0.15, metrics_text, transform=ax.transAxes,
                    ha='center', va='top', fontsize=11, fontweight='semibold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffff', edgecolor='#cccccc', alpha=0.9))
            
    plt.suptitle("Satellite Building Extraction - Multi-Model Inference Report", 
                 fontsize=26, fontweight='bold', y=0.98)
    
    # Add a subtitle/legend
    legend_text = "Masks: Black (Background) / White (Buildings)"
    fig.text(0.5, 0.94, legend_text, ha='center', fontsize=14, style='italic')
    
    # Adjust spacing
    plt.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.06, wspace=0.15, hspace=0.35)
    
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    print(f"Success! Report image saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    if not os.path.exists(TEST_IMG_DIR):
        print(f"Error: {TEST_IMG_DIR} does not exist!")
    else:
        results = run_inference()
        create_report_image(results)
