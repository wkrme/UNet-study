import torch
import os
from torch.utils.data import DataLoader
from loss improt BCEwithDice_Loss
from dataset import UNetDataset
import numpy as np

def test(image_dir, label_dir, model_path,
         batch_size=1, device='cuda'):
    
    all_images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    dataset = UNetDataset(
        image_dir,
        label_dir,
        False
    )
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = UNET_Personal().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"Testing with model: {model_path}")

    criterion = BCEwithDice_Loss()

    bce_scores = []
    dice_scores = []

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            outputs = model(imgs)

            loss, bce_val, dice_val = criterion(outputs, masks)

            bce_scores.append(bce_val.item())
            dice_scores.append(dice_val.item())

    print("Test Result")
    print(f"Average BCE Loss : {np.mean(bce_scores):.4f}")
    print(f"Average Dice Loss: {np.mean(dice_scores):.4f}")
    print(f"Total Loss: {loss}")
    print(f"Average Dice Score: {(1 - np.mean(dice_scores))}")

model_path = "workdir/shlee5_41_0.23922038190066813_0.2463859223657184.pth"
image_dir = "ISBI2016_ISIC_Part1_Training_Data"
label_dir = "ISBI2016_ISIC_Part1_Training_GroundTruth"

test(
    image_dir=image_dir,
    label_dir=label_dir,
    model_path=model_path,
    batch_size=1,
    device="cuda"
)

if __name__ == '__main__':

    model_path = "workdir/shlee5_41_0.23922038190066813_0.2463859223657184.pth"
    image_dir = "ISBI2016_ISIC_Part1_Training_Data"
    label_dir = "ISBI2016_ISIC_Part1_Training_GroundTruth"

    test(image_dir, label_dir, model_path, batch_size=1, device='cuda')