import os, time, torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as TF
from unet import UNet
from data_loader import RootSegmentationDataset
from convert import convert

IMAGE_DIR = '../dataset/images'
MASK_DIR = '../dataset/masks'
BATCH_SIZE = 2
LR = 0.001
NUM_CLASSES = 3
EPOCHS = 20
CHECKPOINT_PATH = "Models/RootSegmentationCheckpoint.pth"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using {DEVICE}")

train_transform = A.Compose([
    A.Resize(300, 400),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ColorJitter(p=0.3),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(300, 400),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
])

full_dataset = RootSegmentationDataset(IMAGE_DIR, MASK_DIR, transform=None)
val_split = 0.01
val_size = int(len(full_dataset) * val_split)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = UNet(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
class_weights = torch.tensor([0.0635, 0.1269, 8.8096], dtype=torch.float32).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.99, patience=15,
)

input_dir = "../dataset/masks"
output_dir = "../dataset/masks"
if any(f.endswith('.json') for f in os.listdir(input_dir)):
    convert(input_dir, output_dir)
    time.sleep(3)
l = []

def train(start_epoch=0, max_epochs=EPOCHS):
    for epoch in range(start_epoch, max_epochs):
        model.train()
        total_loss = 0
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}")

        for i, (imgs, masks) in enumerate(train_loader):
            masks = masks.squeeze(1).long()
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outs = model(imgs)
            if outs.shape[2:] != masks.shape[1:]:
                masks = TF.center_crop(masks.unsqueeze(1), outs.shape[2:]).squeeze(1)
            loss = criterion(outs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            percent = 100.0 * (i + 1) / len(train_loader)
            print(f"  ├─ Batch {i + 1}/{len(train_loader)} ({percent:.1f}%) | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1} Done | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s | LR: {current_lr:.6f}")
        l.append(avg_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                masks = masks.squeeze(1).long()
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                outs = model(imgs)
                if outs.shape[2:] != masks.shape[1:]:
                    masks = TF.center_crop(masks.unsqueeze(1), outs.shape[2:]).squeeze(1)
                loss = criterion(outs, masks)
                val_loss += loss.item()

        scheduler.step(val_loss)

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, CHECKPOINT_PATH)
        print(f"Checkpoint saved at epoch {epoch + 1}")

        torch.save(model.state_dict(), "Models/UNET.pth")
        print(f"Saved model at epoch {epoch + 1}")

if os.path.exists(CHECKPOINT_PATH):
    print("Resuming from checkpoint...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    train(start_epoch, max_epochs=EPOCHS)
else:
    train(0, max_epochs=EPOCHS)

torch.save(model.state_dict(), "Models/UNET.pth")
print("Final model saved to UNET.pth")
print(l)