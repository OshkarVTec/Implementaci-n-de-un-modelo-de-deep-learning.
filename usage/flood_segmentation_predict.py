import torchvision.transforms as transforms
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

MEAN = [0.49858108696687886, 0.4948950363969747, 0.45213825691828213]
STD = [0.16074229459583433, 0.19969003288814977, 0.24751228910355882]


def predict_and_display(image_path, model, device="cpu"):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = (
            torch.argmax(F.softmax(output, dim=1), dim=1).squeeze().cpu().numpy()
        )

    # Post-process the prediction
    pred_mask = Image.fromarray(pred_mask.astype(np.uint8)).resize(
        original_size, Image.NEAREST
    )

    # Display the original image and the prediction
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    # Predicted mask overlayed on the original image
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(pred_mask, cmap="jet", alpha=0.5)
    plt.title("Predicted Mask Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


device = "cpu"
image_path = "usage/flood_image.png"
model = torch.load(
    "usage/efficient_net_flood_segmentation.pt", map_location=torch.device(device)
)
model.eval()
predict_and_display(image_path, model)
