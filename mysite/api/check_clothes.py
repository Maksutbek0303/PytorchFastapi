from fastapi import APIRouter, UploadFile, File, HTTPException
import io, torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from pathlib import Path


classes = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

class CheckImage(nn.Module):
  def __init__(self):
    super().__init__()

    self.first = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(16 * 14 * 14, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
  def forward(self, x):
    x = self.first(x)
    x = self.second(x)
    return x

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

clothing_check_router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / 'all_models'
model_path = MODEL_DIR / 'cloth_model.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CheckImage()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

@clothing_check_router.post('/predict_clothes')
async def check_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(status_code=404, detail="No image")
        img = Image.open(io.BytesIO(image_data))
        image_tenso = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            y_pred = model(image_tenso)
            pred = y_pred.argmax(dim=1).item()
        return {'Answer': classes[pred]}


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
