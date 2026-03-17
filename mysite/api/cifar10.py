from fastapi import APIRouter, UploadFile, File, HTTPException
import torch.nn as nn
import io, torch
from pathlib import Path
from torchvision import transforms
from PIL import Image

classes = ['airplane',
 'automobile',
 'bird',
 'cat',
 'deer',
 'dog',
 'frog',
 'horse',
 'ship',
 'truck']

class CifarClassifaction(nn.Module):
  def __init__(self):
    super().__init__()
    self.first = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
  def forward(self, image):
    image = self.first(image)
    image = self.second(image)
    return image

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

check_cifar_router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / 'all_models'
model_path = MODEL_DIR / 'cifar10_model.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CifarClassifaction()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

@check_cifar_router.post('/predict3')
async def check_image(file: UploadFile = File(...)):
    try:
        image_data = await file.read()

        if not image_data:
            raise HTTPException(status_code=404, detail="No image")
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_tenso = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(image_tenso)
            pred = y_pred.argmax(dim=1).item()
        return {'Answer': classes[pred]}


    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))