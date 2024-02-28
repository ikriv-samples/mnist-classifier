import sys
import torch
from torchvision import transforms
from PIL import Image
from impl.model import CNN
from impl.data import normalization_transform, image_size
from PIL import Image, ImageOps


def infer(image):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalization_transform
    ])
    data = transform(image)
    model = CNN()
    model.load_state_dict(torch.load('nist_model.pt'))
    scores = model(data.unsqueeze(0))
    _, predicted = torch.max(scores, 1)
    return predicted.item()

def inferBuffer(size, buffer):
    image = Image.frombuffer('L', size, buffer)
    return infer(image)

def inferFile(file):
    return infer(ImageOps.invert(Image.open(file).convert('L')))

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python recognize.py <image>")
        sys.exit(1)
    torch.manual_seed(12645734)
    predicted = inferFile(sys.argv[1])
    print(predicted)
