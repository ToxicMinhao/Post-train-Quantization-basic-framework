import torch
from Quantization_model import *
from Quantization_function import *
from Quantization_module import *
from CNNs import *
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


model = torch.load('F:/DRA/prototype_zeroq/mnist_cnn_model_full.pth')
model.eval()

composed = transforms.Compose([transforms.Resize((16, 16)), transforms.ToTensor()])
validation_dataset = MNIST(root='./data', train=False, download=True, transform=composed)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=5000)

quantized_model = quantize_model(model)
quantized_model.eval()

def evaluate_model(model, data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

original_accuracy = evaluate_model(model, validation_loader)
quantized_accuracy = evaluate_model(quantized_model, validation_loader)

print(f'Accuracy of the original model on the test images: {original_accuracy * 100}%')
print(f'Accuracy of the quantized model on the test images: {quantized_accuracy * 100}%')