self.cnn = models.resnet18(weights=None)import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import cv2
import io

app = FastAPI()

#Add route
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html") as f:
        return f.read()

# ==============================
# 🔹 Model Definition
# ==============================
class HybridModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn = models.resnet18(weights=None)
        self.cnn.fc = nn.Identity()

        self.freq_model = models.mobilenet_v2(weights=None)
        self.freq_model.classifier[1] = nn.Identity()

        self.classifier = nn.Linear(512 + 1280, 2)

    def forward(self, spatial, frequency):
        spatial_feat = self.cnn(spatial)
        freq_feat = self.freq_model(frequency)
        combined = torch.cat((spatial_feat, freq_feat), dim=1)
        return self.classifier(combined)

# ==============================
# 🔹 Load Model
# ==============================
device = torch.device("cpu")

model = HybridModel()
model.load_state_dict(torch.load("hybrid_model.pth", map_location=device))
model.eval()

# ==============================
# 🔹 Transform
# ==============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ==============================
# 🔹 Prediction API
# ==============================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Read uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Spatial input
    spatial_img = transform(image).unsqueeze(0).to(device)

    # Frequency input (FFT)
    img_np = np.array(image.convert("L"))

    f = np.fft.fft2(img_np)
    fshift = np.fft.fftshift(f)

    magnitude = 20 * np.log(np.abs(fshift) + 1)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = np.stack((magnitude,)*3, axis=-1)

    freq_img = Image.fromarray(magnitude.astype(np.uint8))
    freq_img = transform(freq_img).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(spatial_img, freq_img)
        probs = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)

    classes = ["fake", "real"]
    confidence = probs[0][predicted.item()].item() * 100

    return {
        "prediction": classes[predicted.item()],
        "confidence": round(confidence, 2)
    }
