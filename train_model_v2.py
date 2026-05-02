import os
import numpy as np
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
import rasterio
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# ======================
# CONFIG (V2 OPTIMIZED)
# ======================
# --- CHOOSE ARCHITECTURE HERE ---
# Options: "unet", "unetplusplus", "linknet", "deeplabv3plus", "manet"
ARCH_CHOICE = "unet"

IMG_SIZE = 256
BATCH_SIZE = 8
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = f"{ARCH_CHOICE}_v2_resnet34"
# Using subdirectories for cleaner multi-model benchmark results
OUTPUT_DIR = os.path.join("results_v2", ARCH_CHOICE)
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, f"best_{MODEL_NAME}.pth")
REPORT_PATH = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_metrics_report.txt")

torch.backends.cudnn.benchmark = True

# ======================
# DATASET
# ======================
class SatelliteDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, filter_empty=False):
        self.samples = []
        self.transform = transform
        file_list = [f for f in os.listdir(img_dir) if f.endswith(".tif")]
        for filename in tqdm(file_list, desc=f"Loading {os.path.basename(img_dir)}"):
            img_path = os.path.join(img_dir, filename); mask_path = os.path.join(mask_dir, filename)
            if not os.path.exists(mask_path): continue
            if filter_empty:
                with rasterio.open(mask_path) as src: m = src.read(1)
                if m is not None and np.sum(m > 0) < 500: continue
            self.samples.append((img_path, mask_path))
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        with rasterio.open(img_path) as src:
            image = src.read(); image = np.transpose(image, (1, 2, 0))
        with rasterio.open(mask_path) as src: mask = src.read(1)

        # V2 Scaling Optimization: Factor 500
        if image.dtype == np.uint16 or image.dtype != np.uint8:
            image = (image.astype('float32') / 500.0 * 255).astype('uint8')
            image = np.clip(image, 0, 255)

        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype('float32')
        if self.transform:
            augmented = self.transform(image=image, mask=mask); image = augmented['image']; mask = augmented['mask']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0; mask = torch.from_numpy(mask).float()
        if mask.ndim == 2: mask = mask.unsqueeze(0)
        return image, mask

# ======================
# AUGMENTATION
# ======================
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
    A.Affine(translate_percent=(0.0625, 0.0625), scale=(0.9, 1.1), rotate=(-15, 15), p=0.5),
    A.GaussNoise(p=0.2), A.RandomBrightnessContrast(p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

IS_CUDA = DEVICE == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=IS_CUDA)

def evaluate_full(model, loader, device, threshold=0.5):
    model.eval(); TP = FP = FN = TN = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device); masks = masks.to(device)
            with torch.amp.autocast("cuda", enabled=IS_CUDA):
                outputs = model(images); preds = torch.sigmoid(outputs)
            preds = (preds > threshold).float(); masks = (masks > 0.5).float()
            TP += (preds * masks).sum().item(); FP += (preds * (1 - masks)).sum().item()
            FN += ((1 - preds) * masks).sum().item(); TN += ((1 - preds) * (1 - masks)).sum().item()
    iou = TP / (TP + FP + FN + 1e-12); f1 = (2 * TP) / (2 * TP + FP + FN + 1e-12)
    return {"iou": iou, "f1": f1, "TP": TP, "FP": FP, "FN": FN, "TN": TN, "precision": TP/(TP+FP+1e-12), "recall": TP/(TP+FN+1e-12), "accuracy": (TP+TN)/(TP+TN+FP+FN+1e-12)}

def get_model(arch):
    if arch == "unet": return smp.Unet
    if arch == "unetplusplus": return smp.UnetPlusPlus
    if arch == "linknet": return smp.Linknet
    if arch == "deeplabv3plus": return smp.DeepLabV3Plus
    if arch == "manet": return smp.MAnet
    raise ValueError(f"Unknown architecture: {arch}")

if __name__ == "__main__":
    train_dataset = SatelliteDataset("dataset_split/train/images", "dataset_split/train/masks", transform=train_transform, filter_empty=True)
    val_dataset = SatelliteDataset("dataset_split/val/images", "dataset_split/val/masks", transform=val_transform)
    test_dataset = SatelliteDataset("dataset_split/test/images", "dataset_split/test/masks", transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model_class = get_model(ARCH_CHOICE)
    model = model_class(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1).to(DEVICE)

    # V2 Loss: Dice + BCE Combo
    dice_loss = smp.losses.DiceLoss(mode='binary')
    bce_loss = nn.BCEWithLogitsLoss()
    def loss_fn(pred, target): return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)

    best_iou = 0; patience = 15; no_improve = 0

    for epoch in range(EPOCHS):
        model.train(); train_loss = 0
        for images, masks in train_loader:
            images = images.to(DEVICE); masks = masks.to(DEVICE)
            with torch.amp.autocast("cuda", enabled=IS_CUDA):
                outputs = model(images); loss = loss_fn(outputs, masks)
            optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); train_loss += loss.item()

        train_loss /= len(train_loader); val_metrics = evaluate_full(model, val_loader, DEVICE); scheduler.step(val_metrics["iou"])

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]; torch.save(model.state_dict(), CHECKPOINT_PATH); no_improve = 0
        else: no_improve += 1

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | IoU: {val_metrics['iou']:.4f} | F1: {val_metrics['f1']:.4f}")
        
        if epoch % 5 == 0:
            with torch.no_grad():
                sample_pred = torch.sigmoid(outputs[0]).cpu().numpy()[0]
                save_path = os.path.join(OUTPUT_DIR, f"{MODEL_NAME}_pred_epoch_{epoch}.png")
                cv2.imwrite(save_path, (sample_pred * 255).astype(np.uint8))

        if no_improve >= patience: print("Early stopping triggered"); break

    if os.path.exists(CHECKPOINT_PATH): model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    train_res = evaluate_full(model, train_loader, DEVICE); val_res = evaluate_full(model, val_loader, DEVICE); test_res = evaluate_full(model, test_loader, DEVICE)
    with open(REPORT_PATH, "w") as f:
        for name, res in [("TRAIN", train_res), ("VALIDATION", val_res), ("TEST", test_res)]:
            f.write(f"\n===== {name} =====\nIoU: {res['iou']:.4f}\nF1 Score: {res['f1']:.4f}\n")
            f.write(f"Precision: {res['precision']:.4f}\nRecall: {res['recall']:.4f}\nAccuracy: {res['accuracy']:.4f}\n")
    print(f"\nV2 Metrics saved to {REPORT_PATH}")
