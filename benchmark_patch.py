import os
import cv2
import time
import torch
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse

# ==============================
# ⚙️ CONFIGURATION
# ==============================
MODEL_DIR = r"D:\Research Project\results_v2"
OUTPUT_DIR = r"D:\Research Project\Benchmark_Results"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 256
THRESHOLD = 0.5
NUM_RUNS = 10  # Number of runs to average latency

MODELS_CONFIG = {
    "unet": {
        "arch": smp.Unet,
        "weights": os.path.join(MODEL_DIR, "unet", "best_unet_v2_resnet34.pth")
    },
    "unetplusplus": {
        "arch": smp.UnetPlusPlus,
        "weights": os.path.join(MODEL_DIR, "unetplusplus", "best_unetplusplus_v2_resnet34.pth")
    },
    "manet": {
        "arch": smp.MAnet,
        "weights": os.path.join(MODEL_DIR, "manet", "best_manet_v2_resnet34.pth")
    },
    "deeplabv3plus": {
        "arch": smp.DeepLabV3Plus,
        "weights": os.path.join(MODEL_DIR, "deeplabv3plus", "best_deeplabv3plus_v2_resnet34.pth")
    },
    "linknet": {
        "arch": smp.Linknet,
        "weights": os.path.join(MODEL_DIR, "linknet", "best_linknet_v2_resnet34.pth")
    }
}

# ==============================
# 🏗️ MODEL LOADER
# ==============================
def load_model(name, config):
    print(f"--- Loading {name}...")
    model = config["arch"](
        encoder_name="resnet34",
        in_channels=3,
        classes=1
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(config["weights"], map_location=DEVICE))
    model.eval()
    return model

# ==============================
# 🔄 PREPROCESSING
# ==============================
transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def preprocess_patch(img_path):
    with rasterio.open(img_path) as src:
        image = src.read()
        image = np.transpose(image, (1, 2, 0)) # To HWC
    
    # Scaling optimization (16-bit to 8-bit using factor 500)
    orig_8bit = (image.astype('float32') / 500.0 * 255).astype('uint8')
    orig_8bit = np.clip(orig_8bit, 0, 255)
    
    # Resize and Transform
    image_resized = cv2.resize(orig_8bit, (IMG_SIZE, IMG_SIZE))
    augmented = transform(image=image_resized)
    input_tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    return input_tensor, image_resized

# ==============================
# 🚀 BENCHMARKING FUNCTION
# ==============================
def benchmark_model(model, input_tensor, name):
    print(f"--- Benchmarking {name}...")
    latencies = []
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_tensor)
    
    # Actual Benchmark
    with torch.no_grad():
        for _ in range(NUM_RUNS):
            start_time = time.time()
            output = model(input_tensor)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            latencies.append((time.time() - start_time) * 1000) # ms
            
        prob = torch.sigmoid(output).cpu().numpy()[0, 0]
        mask = (prob > THRESHOLD).astype(np.uint8)
        
    avg_latency = np.mean(latencies)
    return mask, avg_latency

# ==============================
# 🏁 MAIN
# ==============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark multiple models on a single patch.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input .tif patch")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"ERROR: File {args.input} does not exist.")
        exit(1)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    input_tensor, display_img = preprocess_patch(args.input)
    patch_name = os.path.basename(args.input).split('.')[0]
    
    results = []
    masks = {}
    
    print(f"Starting benchmark for patch: {patch_name}")
    print(f"Device: {DEVICE}")
    print("-" * 40)
    
    for name, config in MODELS_CONFIG.items():
        model = load_model(name, config)
        mask, latency = benchmark_model(model, input_tensor, name)
        
        masks[name] = mask
        results.append({"Model": name, "Latency (ms)": round(latency, 2)})
        
        # Save individual mask
        mask_path = os.path.join(OUTPUT_DIR, f"{patch_name}_{name}_mask.png")
        cv2.imwrite(mask_path, mask * 255)
        
        print(f"DONE: {name} - {latency:.2f} ms")
        del model # Free memory
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
            
    # Save Report
    df = pd.DataFrame(results)
    report_path = os.path.join(OUTPUT_DIR, f"{patch_name}_latency_report.csv")
    df.to_csv(report_path, index=False)
    
    # Visualization
    fig, axes = plt.subplots(1, 6, figsize=(20, 4))
    axes[0].imshow(display_img)
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    for i, (name, mask) in enumerate(masks.items()):
        axes[i+1].imshow(mask, cmap='gray')
        latency = next(r["Latency (ms)"] for r in results if r["Model"] == name)
        axes[i+1].set_title(f"{name}\n({latency}ms)")
        axes[i+1].axis('off')
        
    plt.tight_layout()
    viz_path = os.path.join(OUTPUT_DIR, f"{patch_name}_comparison.png")
    plt.savefig(viz_path)
    plt.close()
    
    print("-" * 40)
    print(f"Summary Table:")
    print(df.to_string(index=False))
    print(f"\nBenchmarking complete! Results saved in: {OUTPUT_DIR}")
    print(f"Visualization: {viz_path}")
    print(f"Report: {report_path}")
