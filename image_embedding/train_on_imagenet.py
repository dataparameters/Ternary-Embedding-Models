import torch
import torch.optim as optim
import torch.nn as nn
from transformers import ViTForImageClassification
from tqdm import tqdm
from utils.get_datasets import imagenet_dataloader
from utils.TnModules import replace_linear


def load_vit_model(model_path, num_labels=100, ignore_mismatched_sizes=True):
    return ViTForImageClassification.from_pretrained(
        model_path, 
        num_labels=num_labels, 
        ignore_mismatched_sizes=ignore_mismatched_sizes
    ).vit


def train_model(train_loader, model1, model2, model, device, epochs=10):
    criterion = nn.MSELoss()
    model1.eval()
    model2.train()  # 只训练 model2

    optimizer = optim.Adam(model2.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)  # 学习率调节器

    for epoch in range(epochs):
        running_loss = 0.0
        model2.to(device)  # 确保 model2 在 GPU 上
        model1.to(device)  # 确保 model1 在 GPU 上

        for images, _ in tqdm(train_loader):
            images = images.to(device)

            optimizer.zero_grad()
            with torch.no_grad():  # model1 输出只作为target
                targets = model1(images).last_hidden_state

            outputs = model2(images).last_hidden_state
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        save_model(model, model2, epoch)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader)}")

    print("training finished！")


def save_model(model, model2, epoch):
    try:
        model2.to('cpu')
        model.vit = model2
        model.save_pretrained(f'Tn_vit_imagenet_{epoch}epoch')
        model2.to(device)
    except Exception as e:
        print(f"can not save the model: {e}")

def main():
    # load imagenet dataset
    train_loader = imagenet_dataloader(
        "/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/datasets/imagenet", 
        batch_size=512, 
        num_workers=4, 
        split='train'
    )

    # load models
    model1 = load_vit_model("/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/models/vit-base-patch16-224")
    model2 = load_vit_model("/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/models/vit-base-patch16-224")
    replace_linear(model2)

    model = ViTForImageClassification.from_pretrained(
        "/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/models/vit-base-patch16-224", 
        num_labels=100, 
        ignore_mismatched_sizes=True
    )

    # freeze model1
    for param in model1.parameters():
        param.requires_grad = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_model(train_loader, model1, model2, model, device, epochs=10)

    # save the final model
    model.vit = model2
    model.to(device)
    model.save_pretrained(f'Tn_vit_imagenet_final')


if __name__ == "__main__":
    main()
