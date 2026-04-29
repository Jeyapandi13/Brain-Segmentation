import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import nn, optim

def main():

    # 🔥 Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ⚡ Faster preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Dataset
    dataset = ImageFolder("dataset", transform=transform)
    print(dataset.class_to_idx)

    # ✅ FIXED loader (no crash)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0   # 🔥 IMPORTANT FIX
    )

    # Model
    from torchvision.models import resnet18, ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # Final layer
    model.fc = nn.Linear(model.fc.in_features, 4)
    model = model.to(device)

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    print("Training Started...")

    EPOCHS = 5

    for epoch in range(EPOCHS):

        running_loss = 0

        for images, labels in loader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), "model/tumor_model.pth")

    print("✅ Model Saved FAST!")

# 🔥 REQUIRED for Windows
if __name__ == "__main__":
    main()
    