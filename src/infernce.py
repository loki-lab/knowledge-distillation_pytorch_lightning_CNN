from lightning_model import LightningModel
from models import VGG16
from utils import get_transforms, load_data
from PIL import Image
import torch

in_chanel = 3
num_classes = 2
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    vgg16_model = VGG16(in_chanels=in_chanel,
                        num_classes=num_classes)
    lightning_model = LightningModel.load_from_checkpoint("checkpoints/best_model.ckpt", model=vgg16_model)
    transforms = get_transforms()["test"]

    image = Image.open("datasets/PetImg/Dog/dog.0.jpg")
    input = transforms(image).unsqueeze(0)
    input = input.to(device)

    output = lightning_model(input)
    print(output)
