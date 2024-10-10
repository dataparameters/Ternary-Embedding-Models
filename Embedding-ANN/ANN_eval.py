import datasets
from mteb.tasks.Retrieval.zho.CMTEBRetrieval import CmedqaRetrieval
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import time
import faiss
from BitBlasModules import replace_linear2bitblas

def generate_embedding(data: dict, tokenizer, model, is_corpus: bool, batch_size=64, device='cuda:0'):
    """
    Generate embeddings for the given data using the tokenizer and model.
    """
    embeddings = {}
    keys = list(data.keys())
    for start_idx in tqdm(range(0, len(keys), batch_size), desc="Generating Embeddings"):
        end_idx = min(start_idx + batch_size, len(keys))
        if is_corpus:
            texts = [data[key]['text'] for key in keys[start_idx:end_idx]]
        else:
            texts = [data[key] for key in keys[start_idx:end_idx]]
        
        # Tokenize and move to device
        tokens = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings_batch = outputs.last_hidden_state[:, 0].cpu().numpy()
        
        # Update the embeddings dictionary
        for idx, key in enumerate(keys[start_idx:end_idx]):
            embeddings[key] = embeddings_batch[idx]

    return embeddings

def precision(qrels, query_keys, retrieved_keys):
    """
    Calculate the precision metric for the retrieval task.
    """
    precisions = []
    for query, retrieved in zip(query_keys, retrieved_keys):
        correct_retrieved = set(qrels.get(query, []))
        retrieved_set = set(retrieved)
        if retrieved_set:
            precision = len(correct_retrieved & retrieved_set) / len(retrieved_set)
        else:
            precision = 0
        precisions.append(precision)
    
    return np.mean(precisions) if precisions else 0

def eval(dataset, qrels, corpus_embeddings, model, tokenizer, device, k=10, batch_size=128,ANN='HNSW'):
    """
    Evaluate the model on the dataset by performing retrieval using FAISS.
    """
    print("Building FAISS Index...")
    start = time.time()

    # Initialize FAISS index
    corpus_array = np.array(list(corpus_embeddings.values()))
    d = corpus_array.shape[1]

    if ANN=='FLAT':
        res = faiss.StandardGpuResources() 
        index = faiss.IndexFlatL2(d)
        index = faiss.index_cpu_to_gpu(res, 0, index)
    if ANN=='IVFLAT':
        res = faiss.StandardGpuResources()
        nlist = 100  
        quantizer = faiss.IndexFlatL2(d) 
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.train(corpus_array)
        index = faiss.index_cpu_to_gpu(res, 0, index)
    if ANN=='LSH':
        res = faiss.StandardGpuResources() 
        index = faiss.IndexLSH(d, 128)
        index = faiss.index_cpu_to_gpu(res, 0, index)
    if ANN=='HNSW':
        index = faiss.IndexHNSWFlat(d, 32)
    
    index.add(corpus_array)

    print(f"FAISS index built in {time.time() - start:.2f} seconds.")
    
    # Prepare queries
    query_keys = list(dataset.queries['dev'].keys())
    paired_keys = []

    print("Evaluating...")
    start = time.time()

    for start_idx in tqdm(range(0, len(query_keys), batch_size), desc="Evaluating"):
        end_idx = min(start_idx + batch_size, len(query_keys))
        batch_keys = query_keys[start_idx:end_idx]
        batch_texts = [dataset.queries['dev'][key] for key in batch_keys]

        # Tokenize and move to device
        tokens = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)

        # Generate embeddings
        with torch.no_grad():
            query_embeddings = model(**tokens).last_hidden_state[:, 0].cpu().numpy()

        # FAISS search
        distances, indices = index.search(query_embeddings, k)

        # Retrieve paired keys
        corpus_keys = list(dataset.corpus['dev'].keys())
        paired_keys.extend([[corpus_keys[i] for i in ids] for ids in indices])

    # Calculate precision
    p = precision(qrels, query_keys, paired_keys)
    
    print(f"Evaluation completed in {time.time() - start:.2f} seconds.")
    print(f"Precision@{k}: {p:.4f}")

    return p


def main(ANN,BITBLAS=False):
    dataset_qrels = datasets.load_dataset('C-MTEB/CmedqaRetrieval-qrels')
    dataset = CmedqaRetrieval()
    dataset.load_data()

    device = 'cuda:0'
    tokenizer = AutoTokenizer.from_pretrained('/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/models/xiaobu-embedding-v2')
    
    if BITBLAS:
        model = AutoModel.from_pretrained('/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/models/xiaobu-embedding-tn').to(device)
        replace_linear2bitblas(model)
    else:
        model = AutoModel.from_pretrained('/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/models/xiaobu-embedding-v2').to(device)

    # Generate corpus embeddings
    start = time.time()
    corpus_embeddings = generate_embedding(dataset.corpus['dev'], tokenizer, model, is_corpus=True)
    print(f"Corpus embeddings generated in {time.time() - start:.2f} seconds.")

    # Prepare qrels
    qp_pairs = defaultdict(list)
    for qid, pid in zip(dataset_qrels['dev']['qid'], dataset_qrels['dev']['pid']):
        qp_pairs[qid].append(pid)

    # Evaluate
    eval(dataset, qp_pairs, corpus_embeddings, model, tokenizer, device,ANN=ANN)


if __name__ == "__main__":
    main(ANN='HNSW',BITBLAS=False)
