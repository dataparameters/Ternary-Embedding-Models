from transformers import AutoTokenizer, AutoModel
from utils.TnModules import replace_linear
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from utils.get_datasets import text_dataloader
from tqdm import tqdm
import torch

def initialize_models(path, device):
    tokenizer = AutoTokenizer.from_pretrained(path)
    modelA = AutoModel.from_pretrained(path).to(device)
    modelB = AutoModel.from_pretrained(path)
    replace_linear(modelB) 
    modelB = modelB.to(device)

    print(modelB)
    
    modelA.eval()
    for param in modelA.parameters():
        param.requires_grad = False

    return tokenizer, modelA, modelB


def train_model(modelA, modelB, tokenizer, trainloader, device, num_epochs, lr=2e-5):
    loss_fn = torch.nn.MSELoss()
    
    for epoch in range(num_epochs):
        optimizer = optim.Adam(modelB.parameters(), lr=lr * (0.2 ** epoch), weight_decay=0)
        modelB.train()

        total_loss = 0
        for inputs in tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            with torch.no_grad():
                targets = modelA(**inputs).last_hidden_state

            outputsB = modelB(**inputs).last_hidden_state
            
            optimizer.zero_grad()
            loss = loss_fn(outputsB, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(trainloader)
        print(f'Epoch {epoch + 1}, Average Loss: {avg_loss:.6f}')

    return modelB

def get_tn_body(path, batch_size=16, num_epochs=2, device='cuda:0'):
    tokenizer, modelA, modelB = initialize_models(path, device)
    
    trainloader = text_dataloader(batch_size)

    modelB = train_model(modelA, modelB, tokenizer, trainloader, device, num_epochs)

    return modelB

def eval(model_name,device,train=False,ternary=False,bitblas=False,bitblas_weight=None,save=False):
    import os
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

    import mteb
    from sentence_transformers import SentenceTransformer


    model = SentenceTransformer(model_name).to(device)

    if train:
        body = get_tn_body(model_name,device=device)
        model._first_module().auto_model = body

    if ternary and not train:
        if bitblas:
            from utils.BitBlasModules import replace_linear2bitblas
            replace_linear2bitblas(model._first_module().auto_model)
            if bitblas_weight is not None:
                from safetensors.torch import load_file
                model._first_module().auto_model.load_state_dict(load_file(bitblas_weight))
                #model._first_module().auto_model.load_state_dict(torch.load(bitblas_weight))
        else:
            replace_linear(model._first_module())

    
    print(model._first_module().auto_model)

    if save:
        if bitblas:
            model.save_pretrained('saved_bitblas')
            #torch.save(model._first_module().auto_model.state_dict(),'weight_bitblas.pth')
            #torch.save(model.state_dict(),'weight_bitblas.pth')
        else:
            model.save_pretrained('saved')

    # task_list = ["TNews","IFlyTek"]

    task_list = ["MultilingualSentiment", "OnlineShopping", "Waimai", "JDReview","MassiveScenarioClassification","MassiveIntentClassification","TNews","IFlyTek",
             "CLSClusteringS2S.v2", "CLSClusteringP2P.v2", "ThuNewsClusteringS2S.v2", "ThuNewsClusteringP2P.v2",
             "Ocnli", "Cmnli",
             "T2Reranking", "MMarcoReranking", "CMedQAv1-reranking", "CMedQAv2-reranking",
             "T2Retrieval", "MMarcoRetrieval", "DuRetrieval", "CovidRetrieval", "CmedqaRetrieval", "EcomRetrieval", "MedicalRetrieval", "VideoRetrieval",
             "ATEC", "BQ", "LCQMC", "PAWSX", "STSB", "AFQMC", "QBQTC","STS22.v2"
             ]

    '''
    task_list = ["AmazonPolarityClassification", "Banking77Classification", "EmotionClassification", "ImdbClassification",
             "ArxivClusteringP2P", "ArxivClusteringS2S", "BiorxivClusteringP2P", "BiorxivClusteringS2S",
             "SprintDuplicateQuestions", "TwitterSemEval2015", "TwitterURLCorpus",
             "AskUbuntuDupQuestions", "SciDocsRR", "StackOverflowDupQuestions",
             "ClimateFEVER", "CQADupstackTexRetrieval", "DBPedia", "FEVER", "FiQA2018", "EcomRetrieval", "MedicalRetrieval", "VideoRetrieval",
             "BIOSSES", "SICK-R", "STS12", "STS13", "STSBenchmark",
             "SummEval"]

    '''
    for t in task_list:
        tasks = mteb.get_tasks(tasks=[t])
        evaluation = mteb.MTEB(tasks=tasks)
        results = evaluation.run(model, output_folder=f"results1", encode_kwargs={'batch_size': 256})

if __name__ == "__main__":
    eval(#model_name = "/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/models/xiaobu-embedding-tn",
         model_name = "/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/ternary-weight-embedding",
         device="cuda:0",
         train=False,
         ternary=True,
         bitblas=True,
         bitblas_weight="/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/ternary-weight-embedding/model.safetensors",
         save=False
         )

#在A800上无法用bitblas跑JDReview