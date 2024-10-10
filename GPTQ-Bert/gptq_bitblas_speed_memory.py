import os

os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_LAUNCH_BLOCKING"]="1"


from functools import reduce
import torch
from transformers import AutoTokenizer, AutoModel
from utils.get_datasets import text_dataloader
import time
from bert import quantize_bert
from utils.BitBlasModules import replace_linear2bitblas

# bitblas.set_log_level("Debug")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str, default='/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/models/xiaobu-embedding-v2',
        help='BERT model to load; pass location of huggingface converted checkpoint.'
        )
    parser.add_argument(
        '--dataset', type=str, choices=['wikitext2','nlizh', 't2rankingzh'], default='t2rankingzh',
        help='Where to extract calibration data from.'
        )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Seed for sampling the calibration data.'
        )
    parser.add_argument(
        '--nsamples', type=int, default=1000,
        help='Number of calibration data samples.'
        )
    parser.add_argument(
        '--percdamp', type=float, default=0.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
        )
    parser.add_argument(
        '--nearest', default=True,
        help='Whether to run the RTN baseline.'
        )
    parser.add_argument(
        '--wbits', type=int, default=2,
        help='Weight bits.'
        )
    parser.add_argument(
        '--act-order', type=int, default=1,
        help='Activation quantization order (1 for constant, 2 for linear).'
        )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to use true sequential quantization.'
        )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize for activation quantization.'
        )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups for activation quantization.'
        )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to use symmetric quantization.'
        )
    parser.add_argument(
        '--save', type=str, default='gptq-xiaoobu-2bit-sample1000-group-1',
        help='Save quantized checkpoint under this name.'
        )

    args = parser.parse_args()

    device="cuda:0"
    model_name = "/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/models/xiaobu-embedding-v2"

    sentences = ["Hey, it is a nice day", "I do not know why it works"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model1 = AutoModel.from_pretrained(model_name)
    model1.load_state_dict(torch.load('/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/models/gptq-xiaobu-2bit-sample1000-group-1'))
    model2 = AutoModel.from_pretrained(model_name)
    model2.load_state_dict(torch.load('/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/models/gptq-xiaobu-2bit-sample1000-group-1'))


    replace_linear2bitblas(model1,args)
    quantize_bert(model2,device,args)

    model1.to(device)
    model2.to(device)
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



    print("GPTQ", get_time(model1))
    print("ref:", get_time(model2))