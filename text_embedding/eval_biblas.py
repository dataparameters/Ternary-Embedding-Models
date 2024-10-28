from transformers import AutoTokenizer, AutoModel
from utils.TnModules import replace_linear
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from utils.get_datasets import text_dataloader
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from utils.BitBlasModules import *
import mteb
from safetensors.torch import load_file

'''
device="cuda:0"
#model_name='/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/saved_bitblas'
model_name='/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/xiaobu-embedding-tn'
model = SentenceTransformer(model_name).to(device)
replace_linear2bitblas(model._first_module().auto_model)
print(model._first_module().auto_model)
model._first_module().auto_model.load_state_dict(load_file("/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/xiaobu-embedding-tn/model.safetensors"))

task_list = [ "OnlineShopping","MultilingualSentiment", "Waimai", "JDReview","MassiveScenarioClassification","MassiveIntentClassification","TNews","IFlyTek","AmazonReviewsClassification",
             "CLSClusteringS2S.v2", "CLSClusteringP2P.v2", "ThuNewsClusteringS2S.v2", "ThuNewsClusteringP2P.v2",
             "CLSClusteringS2S", "CLSClusteringP2P", "ThuNewsClusteringS2S", "ThuNewsClusteringP2P",
             "Ocnli", "Cmnli",
             "T2Reranking", "MMarcoReranking", "CMedQAv1-reranking", "CMedQAv2-reranking",
             "T2Retrieval", "MMarcoRetrieval", "DuRetrieval", "CovidRetrieval", "CmedqaRetrieval", "EcomRetrieval", "MedicalRetrieval", "VideoRetrieval",
             "ATEC", "BQ", "LCQMC", "PAWSX", "STSB", "AFQMC", "QBQTC","STS22.v2","STS22",
             "CMedQAv1", "CMedQAv2"
             ]


for t in task_list:
    tasks = mteb.get_tasks(tasks=[t])
    evaluation = mteb.MTEB(tasks=tasks)
    results = evaluation.run(model, output_folder=f"results1", encode_kwargs={'batch_size': 256})
'''

device="cuda:0"
model_name1 = "/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/models/xiaobu-embedding-tn"
model_name2 = "/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/saved_bitblas"

sentences = ["Hey, it is a nice day", "I do not know why it works"]

tokenizer = AutoTokenizer.from_pretrained(model_name1)

model1 = AutoModel.from_pretrained(model_name1).to(device)
model2 = AutoModel.from_pretrained(model_name2).to(device)

replace_linear2bitblas(model1)
replace_linear2bitblas(model2)

model1.load_state_dict(load_file('/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/saved_bitblas/model.safetensors'))
model2.load_state_dict(load_file('/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/saved_bitblas/model.safetensors'))


print(model1)
print(model2)
model1.eval()
model2.eval()

def compare_model_structure(model1, model2):
    model1_layers = list(model1.named_modules())
    model2_layers = list(model2.named_modules())
    
    if len(model1_layers) != len(model2_layers):
        print("The models have different number of layers.")
        return False
    
    for (name1, layer1), (name2, layer2) in zip(model1_layers, model2_layers):
        if name1 != name2 or type(layer1) != type(layer2):
            print(f"Layer mismatch: {name1} vs {name2}")
            return False
    
    print("The model structures are identical.")
    return True

compare_model_structure(model1, model2)

state_dict1 = model1.state_dict()
state_dict2 = model2.state_dict()
for key in state_dict1:
    if not torch.allclose(state_dict1[key], state_dict2[key], atol=1e-6):
        print(f"Mismatch found in layer: {key}")
    if state_dict1[key].dtype!=state_dict2[key].dtype:
        print(f"Mismatch found in dtype: {key}")

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
