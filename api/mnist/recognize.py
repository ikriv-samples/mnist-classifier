import sys
import torch
from torchvision import transforms
from impl.model import CNN
from impl.data import normalization_transform, image_size
from PIL import Image, ImageOps
import pathlib

def _getModelPath():
    return pathlib.Path(__file__).parent.resolve() / 'nist_model.pt'

def infer(image):
    image = ImageOps.invert(image.convert('L'))
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        normalization_transform
    ])
    data = transform(image)
    model = CNN()
    model.load_state_dict(torch.load(_getModelPath()))
    model.eval()
    scores = model(data.unsqueeze(0))
    _, predicted = torch.max(scores, 1)
    return predicted.item()

def inferBuffer(size, buffer):
    image = Image.new(mode='L', size=size)
    image.putdata(buffer)
    return infer(image)

def inferFile(file):
    return infer(Image.open(file).convert('L'))

def inferDataUrl(dataUrl):
    import re
    import base64
    import io
    match = re.match(r'data:image/png;base64,(.*)', dataUrl)
    if not match:
        raise ValueError("Invalid image data; expected 'data:image/png;base64,<...data...>', got '{dataUrl[:50]}...'")
    image = Image.open(io.BytesIO(base64.b64decode(match.group(1))))
    image = image.convert('L')
    image = image.resize((28, 28), Image.LANCZOS)
    return infer(image)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python recognize.py <image>")
        sys.exit(1)
    torch.manual_seed(12645734)
    predicted = inferFile(sys.argv[1])
    print(predicted)
