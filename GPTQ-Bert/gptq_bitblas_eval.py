import os

os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_LAUNCH_BLOCKING"]="1"


from functools import reduce
import torch
from transformers import AutoTokenizer, AutoModel
from utils.BitBlasModules import replace_linear2bitblas
import mteb
from sentence_transformers import SentenceTransformer

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
    model = SentenceTransformer(model_name).to(device)

    model._first_module().auto_model.load_state_dict(torch.load('/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/models/gptq-xiaobu-2bit-sample1000-group-1'))

    replace_linear2bitblas(model._first_module().auto_model,args)

    model.to(device)
    print(model._first_module().auto_model)

    task_list = ["MultilingualSentiment", "OnlineShopping", "Waimai", "JDReview",
             "CLSClusteringS2S.v2", "CLSClusteringP2P.v2", "ThuNewsClusteringS2S.v2", "ThuNewsClusteringP2P.v2",
             "Ocnli", "Cmnli",
             "T2Reranking", "MMarcoReranking", "CMedQAv1-reranking", "CMedQAv2-reranking",
             "T2Retrieval", "MMarcoRetrieval", "DuRetrieval", "CovidRetrieval", "CmedqaRetrieval", "EcomRetrieval", "MedicalRetrieval", "VideoRetrieval",
             "ATEC", "BQ", "LCQMC", "PAWSX", "STSB", "AFQMC", "QBQTC"]
    for t in task_list:
        tasks = mteb.get_tasks(tasks=[t])
        evaluation = mteb.MTEB(tasks=tasks)
        results = evaluation.run(model, output_folder=f"results1", encode_kwargs={'batch_size': 256})
