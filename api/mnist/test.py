import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
from impl.data import test_loader
from impl.model import CNN

def inferImpl(model, inputs):

    # Convert tensors to Variables (for autograd)
    X_batch = Variable(inputs, volatile=True, requires_grad=False)

    # Forward pass
    scores = model(X_batch)  # logits

    # Accuracy
    score, predicted = torch.max(scores, 1)

    return predicted

def infer():
    model = CNN()
    model.load_state_dict(torch.load('nist_model.pt'))

    batch = next(iter(test_loader))
    samples = batch[0][:5]
    print(samples[0])
    y_preds = inferImpl(model, samples)
    for i, sample in enumerate(samples):
        plt.subplot(1, 5, i+1)
        plt.title('prediction: %i'% y_preds[i].item())
        plt.imshow(sample.numpy().reshape((28,28)))
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    infer()
