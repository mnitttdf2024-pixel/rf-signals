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
# ### NEW ###
# Import from Hugging Face transformers
# ==============================
try:
    from transformers import AutoModelForImageClassification, AutoImageProcessor
except ImportError:
    raise ImportError("Install Hugging Face transformers: pip install transformers")


# ==============================
# 1. Dataset (Your class is great, no changes needed)
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
                # Basic check for image files
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
            # Return a dummy image and label
            return torch.randn(3, 224, 224), -1 # ### MODIFIED ### to 224x224

        if self.transform:
            image = self.transform(image)

        # Handle cases where image loading failed
        if label == -1:
            return image, torch.tensor(-1)  # Return dummy tensor

        return image, label


# ==============================
# 2. Model
# ==============================
# ### REMOVED ###
# The VimForClassification class is no longer needed.
# We will load the model directly in main().

# ==============================
# 3. Train & Evaluate
# ==============================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(loader, desc="Training", leave=False):
        # Handle dummy images from loading errors
        if -1 in labels:
            images = images[labels != -1]
            labels = labels[labels != -1]
            if images.size(0) == 0:
                continue

        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # ### MODIFIED ###
        # Hugging Face models output a class with logits
        outputs = model(images).logits 
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    # ### MODIFIED ### Robust check for empty loader
    if len(loader.dataset) == 0:
        return 0.0
    
    # Calculate loss based on number of items *processed*
    processed_items = len(loader.dataset) - (labels == -1).sum().item()
    if processed_items == 0:
        return 0.0
        
    return total_loss / processed_items


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            # Handle dummy images from loading errors
            if -1 in labels:
                images = images[labels != -1]
                labels = labels[labels != -1]
                if images.size(0) == 0:
                    continue

            images, labels = images.to(device), labels.to(device)
            
            # ### MODIFIED ###
            # Hugging Face models output a class with logits
            outputs = model(images).logits
            
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    if total == 0: # Check if any valid items were processed
        return 0.0, 0.0, [], []  # Avoid division by zero

    acc = correct / total
    loss_avg = total_loss / total # Average loss per item
    return loss_avg, acc, labels_all, preds_all


# ==============================
# 4. Main
# ==============================
def main():
    # --- Config ---
    dataset_root = 'dataset'  # <-- Make sure this folder exists
    results_dir = 'vim_results_transfer_learning' # <-- Changed output dir
    os.makedirs(results_dir, exist_ok=True)

    # --- Hyperparameters (from your file) ---
    batch_size = 32
    num_epochs = 1
    lr = 1e-4
    patience = 10
    weight_decay = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # --- ### NEW ### Model Loading from Hugging Face ---
    MODEL_NAME = "google/vit-base-patch16-224-in21k"
    print(f"Loading pre-trained model: {MODEL_NAME}")
    
    # The processor handles image transformations (size, normalization)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME, 
        trust_remote_code=True
    )
    
    # Get model's expected input size and normalization stats
    image_size = processor.size["height"]
    image_mean = processor.image_mean
    image_std = processor.image_std
    
    print(f"Model loaded. Input size set to: {image_size}x{image_size}")

    # --- ### MODIFIED ### Data Augmentation ---
    # Transforms MUST match the pre-trained model's expectations
    
    train_transform = transforms.Compose([
        # We must resize to what the model expects
        transforms.Resize((image_size, image_size)), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std)
    ])
    
    val_transform = transforms.Compose([
        # Validation just needs resize, ToTensor, and Normalize
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std)
    ])
    # --- ** END MODIFICATIONS ** ---

    # --- ### MODIFIED ### Dataset Splitting ---
    # This is a robust way to apply different transforms to train/val
    # after a random split.
    
    # 1. Create two dataset instances (one for train, one for val)
    #    They point to the same data but have different transform pipelines
    dataset_for_train = CustomDataset(dataset_root, transform=train_transform)
    dataset_for_val = CustomDataset(dataset_root, transform=val_transform)

    if len(dataset_for_train) == 0:
        print(f"Error: No images found in {dataset_root}. Please check.")
        return

    num_classes = len(dataset_for_train.classes)
    print(f"Detected {len(dataset_for_train)} images in {num_classes} classes: {dataset_for_train.classes}")

    # 2. Perform the split on one dataset to get indices
    train_size = int(0.8 * len(dataset_for_train))
    val_size = len(dataset_for_train) - train_size
    
    # Use a fixed generator for reproducible splits
    train_subset, val_subset = random_split(
        dataset_for_train, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42) # For reproducibility
    )

    # 3. CRITICAL: Point the val_subset to use the val_dataset instance
    #    This ensures it uses the validation transforms
    val_subset.dataset = dataset_for_val
    
    # Now train_subset uses train_transform, and val_subset uses val_transform

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, num_workers=4)
    # --- ** END MODIFICATIONS ** ---


    # --- ### NEW ### Freeze Backbone & Replace Head ---
    print("Adapting model for transfer learning...")
    
    # 1. Freeze the backbone (the 'mambavision' part)
    # 1. Freeze the backbone (the 'vit' part)
    for param in model.vit.parameters():
        param.requires_grad = False
    
    # 2. Get the number of input features for the classifier
    in_features = model.classifier.in_features
    
    # 3. Replace the classifier head with a new, unfrozen one
    model.classifier = nn.Linear(in_features, num_classes)
    
    model.to(device) # Move the *entire* model to the device
    
    # --- ** END NEW ** ---

    criterion = nn.CrossEntropyLoss()
    
    # --- ### MODIFIED ### Optimizer ---
    # We ONLY optimize the parameters of the new classifier
    print("Optimizing ONLY the new classifier head.")
    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), # <-- This is the key change
        lr=lr,
        weight_decay=weight_decay
    )
    # --- ** END MODIFICATIONS ** ---

    # --- Training Loop (Your excellent loop is unchanged) ---
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

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        metrics_log.append((epoch+1, train_loss, val_loss, val_acc))

        print(
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(
                results_dir, "best_vim_model.pth"))
            print("** Saved new best model **")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs with no improvement.")
                break

    # --- Logging & Plotting (Your code is unchanged) ---
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
    plt.title("Training and Validation Loss (Transfer Learning Vim)")
    loss_plot_path = os.path.join(results_dir, "loss_curve.png")
    plt.savefig(loss_plot_path)
    print(f"Saved: {loss_plot_path}")
    plt.close()

    # --- Confusion Matrix ---
    cm = confusion_matrix(true_labels, preds)
    # Fix for divide-by-zero if a class has 0 true samples in val set
    cm_sum = cm.sum(axis=1)[:, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_percent = np.nan_to_num(cm.astype(float) / cm_sum * 100)

    plt.figure(figsize=(max(6, num_classes), max(5, num_classes * 0.8)))
    plt.imshow(cm_percent, cmap='Blues',
               interpolation='nearest', vmin=0, vmax=100)
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
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Saved: {cm_path}")
    plt.close()

    # --- Save Metrics Log ---
    metrics_path = os.path.join(results_dir, "training_metrics.csv")
    with open(metrics_path, "w") as f:
        f.write("Epoch,Train_Loss,Val_Loss,Val_Accuracy\n")
        for e, tr, vl, acc in metrics_log:
            f.write(f"{e},{tr:.6f},{vl:.6f},{acc:.6f}\n")
    print(f"Saved: {metrics_path}")

    print(f"\nAll results saved under: {os.path.abspath(results_dir)}")


if __name__ == "__main__":
    main()