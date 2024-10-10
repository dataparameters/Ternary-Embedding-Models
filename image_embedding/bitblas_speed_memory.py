import os

os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import bitblas
import torch
import torch.nn as nn
from utils.TnModules import replace_linear
from utils.BitBlasModules import replace_linear2bitblas
from transformers import ViTForImageClassification, ViTImageProcessor
import time
from utils.get_datasets import *
from tqdm import tqdm


if __name__ == "__main__":  
    device="cuda:0"
    model1 = ViTForImageClassification.from_pretrained("/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/models/Tn_vitbase_16-224_imagenet10epoch", num_labels=100,ignore_mismatched_sizes=True)
    model2 = ViTForImageClassification.from_pretrained("/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/models/Tn_vitbase_16-224_imagenet10epoch", num_labels=100,ignore_mismatched_sizes=True)

    model1.to(device)
    model2.to(device)

    replace_linear(model1.vit)
    replace_linear2bitblas(model2.vit)

    print(model1)
    print(model2)

    def get_size(model,name):
        torch.save(model.state_dict(),name)
        size = os.path.getsize(name)
        return size / (1024 * 1024)

    print(f"bitblas size: {get_size(model2,'ref.pth'):.2f} MB")
    print(f"ref size: {get_size(model1,'bitblas.pth'):.2f} MB")


    def get_time(model):
        trainloader = imagenet_dataloader("/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/datasets/imagenet",256,4,split='val')
        start = time.time()
        for epoch in range(1):
            for inputs,_ in tqdm(trainloader):
                inputs = inputs.to(device)
                with torch.no_grad():
                    model(inputs)

        end = time.time()
        return end - start

    print("bitblas speed", get_time(model2))
    print("ref speed:", get_time(model1))