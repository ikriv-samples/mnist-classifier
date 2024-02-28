from impl.data import raw_test
from torchvision import transforms as t

def row(img):
    return img[0][7][:14].tolist()

batch, _ = next(iter(raw_test))
image = batch[0]
print(row(image))

t1 = t.ToTensor()(image[0].unsqueeze(2).numpy())*255
print(t1.shape)
print(row(t1))
