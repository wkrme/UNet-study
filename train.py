import os
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from torch.utils.data import random_split, DataLoader, Subset

from model import UNet
from dataset import UNetDataset
from loss import BCELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Config #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 50
batch_size = 4
learning_rate = 1e-3
save_path = "workdir"
os.makedirs(save_path, exist_ok=True)

# DataLoader #
img_path = 'ISBI2016_ISIC_Part1_Training_Data/ISBI2016_ISIC_Part1_Training_Data'
label_path = 'ISBI2016_ISIC_Part1_Training_Data/ISBI2016_ISIC_Part1_Training_GroundTruth'

dataset = UNetDataset(img_path, label_path)

train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size

train_dataset, valid_subset = random_split(dataset, [train_size, val_size], torch.Generator().manual_seed(0))
valid_dataset = UNetDataset(img_path, label_path, False)
valid_dataset = Subset(valid_dataset, valid_subset.indices) # Validation Dataset에는 elastic deformation 적용 X 위해

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

# Model #
model = UNet().to(device)

# Loss #
bceloss = BCELoss()

# Optimizer #
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.99)

# Scheduler #
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min', # loss 감소 기준 (max면 metric 증가 기준)
    factor=0.5, # lr 감소 비율
    patience=2, # 개선 없으면 기다릴 epoch 수
    min_lr=1e-6 # lr 최소값 제한
)

# Train #
for epoch in range(1, num_epochs + 1):
    model.train()
    train_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        labels = labels[:, :, :outputs.size()[2], :outputs.size()[3]]

        loss = bceloss(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Iter [{i+1}/{len(train_loader)}], '
                  f'Loss: {train_loss/(i+1):.4f}')

    train_loss /= len(train_loader)

    # Validation #
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            labels = labels[:, :, :outputs.size()[2], :outputs.size()[3]]

            loss = bceloss(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    scheduler.step(val_loss)

    print(f'Epoch [{epoch}/{epoch}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # checkpoint #
    torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}_{train_loss}_{val_loss}.pth'))
