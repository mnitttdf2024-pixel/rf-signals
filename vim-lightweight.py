import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import datetime

# ==============================
# 1. Import Vision Mamba
# ==============================
try:
    # We will use the 'Vim' block from the package you already have
    from vision_mamba import Vim 
except ImportError:
    raise ImportError("Install Vision Mamba: pip install vision-mamba")


# ==============================
# 2. Dataset (Your class)
# ==============================
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.images = []
        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(class_dir):
                print(f"Warning: Skipping {class_dir}, not a directory.")
                continue

            for fname in os.listdir(class_dir):
                path = os.path.join(class_dir, fname)
                if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.images.append((path, self.class_to_idx[cls]))
                else:
                    print(f"Warning: Skipping non-image file: {path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.randn(3, 128, 128), -1 # Use 128x128

        if self.transform:
            image = self.transform(image)

        if label == -1:
            return image, torch.tensor(-1) 

        return image, label


# ==============================
# 3. Model
# ==============================
class UltraLightVim(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        # --- Configuration for a VERY SMALL model ---
        self.vim = Vim(
            dim=96,              # Embedding dimension (very small)
            image_size=128,      # Your image size
            patch_size=16,       # 16x16 patches (gives 8x8=64 patches)
            depth=6,             # Number of Mamba blocks (very few)
            d_state=16,          # Mamba state dimension (standard)
            num_classes=num_classes,

            # --- ### FIX ### ---
            dt_rank=6,           # 96 / 16 = 6
            dim_inner=96,        # <-- FIX: Set dim_inner == dim
        )

    def forward(self, x):
        return self.vim(x)

# ==============================
# 4. Train & Evaluate (Your functions)
# ==============================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        if -1 in labels:
            images = images[labels != -1]
            labels = labels[labels != -1]
            if images.size(0) == 0:
                continue

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    if len(loader.dataset) == 0:
        return 0.0
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            if -1 in labels:
                images = images[labels != -1]
                labels = labels[labels != -1]
                if images.size(0) == 0:
                    continue

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    if total == 0:
        return 0.0, 0.0, [], []  # Avoid division by zero

    acc = correct / total
    loss_avg = total_loss / total
    return loss_avg, acc, labels_all, preds_all


# ==============================
# 5. Main
# ==============================
def main():
    # --- Config ---
    dataset_root = 'dataset'  # <-- Make sure this folder exists
    results_dir = 'vim_results_ultralight_01' # <-- New output dir
    os.makedirs(results_dir, exist_ok=True)

    # --- Hyperparameters ---
    batch_size = 32        
    num_epochs = 200       # Training from scratch needs a lot of epochs
    lr = 5e-4              
    patience = 25          # Increased patience
    weight_decay = 5e-3    # Strong regularization (weight decay)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Data Augmentation (CRITICAL for training from scratch) ---
    # We need to be very aggressive with augmentation
    train_transform = transforms.Compose([
        transforms.TrivialAugmentWide(), # Strong, modern augmentation
        transforms.Resize(128), # TrivialAugment might change size
        transforms.RandomCrop(128), # Ensure 128x128
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], # Standard ImageNet stats
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(128), 
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # --- Dataset Splitting ---
    dataset_for_train = CustomDataset(dataset_root, transform=train_transform)
    dataset_for_val = CustomDataset(dataset_root, transform=val_transform)

    if len(dataset_for_train) == 0:
        print(f"Error: No images found in {dataset_root}. Please check.")
        return

    num_classes = len(dataset_for_train.classes)
    print(f"Detected {len(dataset_for_train)} images in {num_classes} classes: {dataset_for_train.classes}")

    train_size = int(0.8 * len(dataset_for_train))
    val_size = len(dataset_for_train) - train_size
    
    train_subset, val_subset = random_split(
        dataset_for_train, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    val_subset.dataset = dataset_for_val
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=4)
    # --- End Split ---

    # --- ### MODEL INITIALIZATION ### ---
    print("Initializing 'Ultra-Light' Vision Mamba (from scratch)...")
    model = UltraLightVim(num_classes=num_classes).to(device)
    
    # --- Count Parameters ---
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total_params / 1_000_000:.2f}M")
    # ---

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=weight_decay
    )
    
    # Add a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # --- Training Loop ---
    best_acc = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    metrics_log = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, true_labels, preds = evaluate(
            model, val_loader, criterion, device)
        
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        metrics_log.append((epoch+1, train_loss, val_loss, val_acc))

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.1e}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(
                results_dir, "best_ultralight_vim_scratch.pth"))
            print("** Saved new best model **")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs with no improvement.")
                break

    # --- Logging & Plotting ---
    if not true_labels or not preds:
        print("Evaluation did not run. Skipping plots and metrics.")
        print(f"\nAll results saved under: {os.path.abspath(results_dir)}")
        return

    # --- Save Loss Plot ---
    plt.figure()
    plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss (Ultra-Light ViM)")
    loss_plot_path = os.path.join(results_dir, "loss_curve_scratch.png")
    plt.savefig(loss_plot_path)
    print(f"Saved: {loss_plot_path}")
    plt.close()

    # --- Confusion Matrix ---
    cm = confusion_matrix(true_labels, preds)
    cm_sum = cm.sum(axis=1)[:, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_percent = np.nan_to_num(cm.astype(float) / cm_sum * 100)

    plt.figure(figsize=(max(6, num_classes), max(5, num_classes * 0.8)))
    plt.imshow(cm_percent, cmap='Blues', interpolation='nearest', vmin=0, vmax=100)
    plt.title("Confusion Matrix (%)")
    plt.colorbar()
    ticks = np.arange(num_classes)
    plt.xticks(ticks, dataset_for_train.classes, rotation=45, ha="right")
    plt.yticks(ticks, dataset_for_train.classes)

    threshold = cm_percent.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)",
                     ha='center', va='center',
                     color="white" if cm_percent[i, j] > threshold else "black",
                     fontsize=8)

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrix_scratch.png")
    plt.savefig(cm_path)
    print(f"Saved: {cm_path}")
    plt.close()

    # --- Save Metrics Log ---
    metrics_path = os.path.join(results_dir, "training_metrics_scratch.csv")
    with open(metrics_path, "w") as f:
        f.write("Epoch,Train_Loss,Val_Loss,Val_Accuracy\n")
        for e, tr, vl, acc in metrics_log:
            f.write(f"{e},{tr:.6f},{vl:.6f},{acc:.6f}\n")
    print(f"Saved: {metrics_path}")

    print(f"\nAll results saved under: {os.path.abspath(results_dir)}")


if __name__ == "__main__":
    main()