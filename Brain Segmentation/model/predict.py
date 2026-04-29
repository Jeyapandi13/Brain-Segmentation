import torch
import torchvision.models as models
from torchvision import transforms
from torch import nn
from PIL import Image
import numpy as np
import cv2
import os

# 🔥 Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

DISPLAY_NAMES = {
    'glioma': 'Glioma',
    'meningioma': 'Meningioma',
    'notumor': 'No Tumor',
    'pituitary': 'Pituitary'
}

# Colors
CLASS_COLORS = {
    "Glioma": (0, 0, 255),
    "Meningioma": (0, 165, 255),
    "No Tumor": (0, 255, 0),
    "Pituitary": (255, 0, 255)
}

# Risk
RISK_MULTIPLIER = {
    "Glioma": 1.0,
    "Meningioma": 0.7,
    "No Tumor": 0.0,
    "Pituitary": 0.5
}

# 🔥 FAST transform (must match training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),   # 🔥 IMPORTANT CHANGE
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load model
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)

    model_path = os.path.join(os.path.dirname(__file__), "tumor_model.pth")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()


def predict_tumor(image_path, segmented_folder, filename):

    original_image = Image.open(image_path).convert("RGB")
    input_tensor = transform(original_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][pred_idx].item()

    raw_class = CLASSES[pred_idx]
    tumor_type = DISPLAY_NAMES[raw_class]
    tumor = "No" if raw_class == "notumor" else "Yes"

    base_risk = confidence * 100
    risk = round(base_risk * RISK_MULTIPLIER.get(tumor_type, 1.0), 1)

    all_probs = {
        DISPLAY_NAMES[CLASSES[i]]: probabilities[0][i].item() * 100
        for i in range(len(CLASSES))
    }

    segmented_filename = create_visualization(
        image_path,
        segmented_folder,
        filename,
        tumor_type,
        confidence,
        all_probs,
        tumor
    )

    return tumor, tumor_type, risk, segmented_filename


def create_visualization(image_path, segmented_folder, filename,
                         tumor_type, confidence, all_probs, tumor_detected):

    image = cv2.imread(image_path)
    if image is None:
        return f"seg_{filename}"

    height, width = image.shape[:2]

    canvas = np.zeros((height + 120, width, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)
    canvas[0:height, 0:width] = image

    if tumor_detected == "Yes":
        heatmap = create_attention_heatmap(image, tumor_type)
        blended = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
        canvas[0:height, 0:width] = blended

        color = CLASS_COLORS.get(tumor_type, (0, 0, 255))

        margin = int(min(width, height) * 0.15)
        x1, y1 = margin, margin
        x2, y2 = width - margin, height - margin

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)

        cv2.rectangle(canvas, (0, 0), (width, 35), (0, 0, 180), -1)
        cv2.putText(canvas, f"TUMOR: {tumor_type}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    else:
        green = np.zeros_like(image)
        green[:] = (0, 100, 0)
        canvas[0:height, 0:width] = cv2.addWeighted(image, 0.9, green, 0.1, 0)

        cv2.rectangle(canvas, (0, 0), (width, 35), (0, 150, 0), -1)
        cv2.putText(canvas, "NO TUMOR",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Probabilities
    y_start = height + 10
    for i, (cls, prob) in enumerate(sorted(all_probs.items(),
                                           key=lambda x: x[1],
                                           reverse=True)):

        y = y_start + i * 25
        bar_width = int((prob / 100) * (width - 20))

        cv2.rectangle(canvas, (10, y), (10 + bar_width, y + 20),
                      CLASS_COLORS.get(cls, (128, 128, 128)), -1)

        cv2.putText(canvas, f"{cls}: {prob:.1f}%",
                    (15, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    segmented_filename = f"seg_{filename}"
    cv2.imwrite(os.path.join(segmented_folder, segmented_filename), canvas)

    return segmented_filename


def create_attention_heatmap(image, tumor_type):
    h, w = image.shape[:2]

    y, x = np.ogrid[:h, :w]
    center_x, center_y = w // 2, h // 2

    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    heatmap = 1 - (dist / dist.max())

    heatmap = cv2.GaussianBlur(heatmap.astype(np.float32), (31, 31), 0)
    heatmap = (heatmap * 255).astype(np.uint8)

    color = CLASS_COLORS.get(tumor_type, (0, 0, 255))
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(3):
        colored[:, :, i] = (heatmap * (color[i] / 255)).astype(np.uint8)

    return colored


def predict_tumor_simple(image_path):

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][idx].item()

    raw = CLASSES[idx]
    tumor_type = DISPLAY_NAMES[raw]
    tumor = "No" if raw == "notumor" else "Yes"

    risk = round(confidence * 100 * RISK_MULTIPLIER.get(tumor_type, 1.0), 1)

    return tumor, tumor_type, risk