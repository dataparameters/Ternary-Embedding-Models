import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from utils.get_datasets import imagenet_dataloader,cifar10_dataloader,cifar100_dataloader
import torch.cuda.amp as amp 
from utils.TnModules import replace_linear
from utils.BitBlasModules import replace_linear2bitblas


# Initialize Dataloaders for training and testing
def get_dataloaders(data_name, batch_size=512, num_workers=4):
    if data_name=='imagenet':
        train_loader = imagenet_dataloader(
            "/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/datasets/imagenet", batch_size, num_workers, split='train')
        test_loader = imagenet_dataloader(
            "/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/datasets/imagenet", batch_size, num_workers, split='val')
    if data_name=='cifar10':
        train_loader=cifar10_dataloader(batch_size,num_workers,split='train')
        test_loader=cifar10_dataloader(batch_size,num_workers,split='test')
    if data_name=='cifar100':
        train_loader=cifar100_dataloader(batch_size,num_workers,split='train')
        test_loader=cifar100_dataloader(batch_size,num_workers,split='test')
    return train_loader, test_loader


# Load pre-trained model and freeze parameters
def load_model(path,num_labels=1000, device='cuda:0',ternary=False):
    model = ViTForImageClassification.from_pretrained(
        path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    if ternary: # use ternary weight model
        replace_linear2bitblas(model.base_model)
    print(model)
    # Freeze all parameters except for the classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    model.to(device)
    return model


def train_model(model, train_loader, epochs=1, initial_lr=1e-3, device='cuda:0'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=initial_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            outputs = model(images).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()  # Update learning rate
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")
        torch.cuda.empty_cache()  # Optional, once per epoch to free memory


# Evaluate the model
def evaluate_model(model, test_loader, device='cuda:0'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images).logits
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


# Save the trained model
def save_model(model, accuracy):
    model.save_pretrained(f'cifar100_vit_imagnet_{accuracy}')


# Main function to run everything
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders('cifar10',batch_size=512, num_workers=4)
    
    path="/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/models/Tn_vitbase_16-224_imagenet10epoch"
    model = load_model(path, num_labels=10, device=device,ternary=True)

    # Train the classifier and Evaluate the model
    train_model(model, train_loader, epochs=10, initial_lr=1e-3, device=device)
    accuracy = evaluate_model(model, test_loader, device=device)

    # Save the model
    save_model(model, accuracy)

    print("Training and evaluation complete!")


if __name__ == "__main__":
    main()
