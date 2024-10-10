import torch
from transformers import ViTForImageClassification
from tqdm import tqdm
from utils.get_datasets import imagenet_dataloader
from utils.BitBlasModules import replace_linear2bitblas

def load_model(model_path, num_labels=1000, device='cuda:0', ignore_mismatch=True):
    model = ViTForImageClassification.from_pretrained(model_path, num_labels=num_labels, ignore_mismatched_sizes=ignore_mismatch)
    print('It may take some time for the first run.')
    replace_linear2bitblas(model.base_model) 
    model.to(device)
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def main():
    model_path = "/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/models/imagenet_vit_74.774"
    dataset_path = "/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/datasets/imagenet"
    device='cuda:0'
    test_loader = imagenet_dataloader(dataset_path, batch_size=512, num_workers=4, split='val')

    model = load_model(model_path, num_labels=1000, device=device)

    accuracy = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.2f}%")
    
if __name__ == "__main__":
    main()