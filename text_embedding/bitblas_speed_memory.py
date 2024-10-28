import os

os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

import torch
from transformers import AutoTokenizer, AutoModel
from utils.TnModules import replace_linear
from utils.BitBlasModules import replace_linear2bitblas
from utils.get_datasets import text_dataloader
import time

if __name__ == "__main__":  
    
    device="cuda:0"
    model_name = "/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/models/xiaobu-embedding-tn"
    #model_name='/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/models/stella-base-zh-tn'

    sentences = ["Hey, it is a nice day", "I do not know why it works"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model1 = AutoModel.from_pretrained(model_name).to(device)
    model2 = AutoModel.from_pretrained(model_name).to(device)

    replace_linear2bitblas(model1)
    replace_linear(model2)

    print(model1)
    print(model2)
    model1.eval()
    model2.eval()

    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    for _ in encoded_input:
        encoded_input[_] = encoded_input[_].to(device)

    with torch.no_grad():
        model_output1 = model1(**encoded_input)
        sentence_embeddings1 = model_output1[0][:, 0]
        model_output2 = model2(**encoded_input)
        sentence_embeddings2 = model_output2[0][:, 0]

    sentence_embeddings1 = torch.nn.functional.normalize(sentence_embeddings1, p=2, dim=1)
    print("Sentence embeddings 1:", sentence_embeddings1)
    sentence_embeddings2 = torch.nn.functional.normalize(sentence_embeddings2, p=2, dim=1)
    print("Sentence embeddings 2:", sentence_embeddings2)

    from tqdm import tqdm

    def get_size(model,name):
        torch.save(model.state_dict(),name)
        size = os.path.getsize(name)
        return size / (1024 * 1024)

    print(f"bitblas size: {get_size(model2,'ref.pth'):.2f} MB")
    print(f"ref size: {get_size(model1,'bitblas.pth'):.2f} MB")

    def get_time(model):
        trainloader = text_dataloader(batch_size=512)
        start = time.time()
        for epoch in range(1):
            for inputs in tqdm(trainloader):
                inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = inputs.to(device)
                with torch.no_grad():
                    model(**inputs)

        end = time.time()
        return end - start

    print("ref:", get_time(model2))
    print("bitblas", get_time(model1))