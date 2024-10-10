import time
import torch
import torch.nn as nn
from transformers import AutoModel
import mteb

from GPTQ.gptq import *
from GPTQ.modelutils import *
from GPTQ.datautils import *

def get_bert(model_name):
    model = AutoModel.from_pretrained(model_name, torch_dtype='auto')
    return model

@torch.no_grad()
def bert_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.encoder.layer

    model.embeddings = model.embeddings.to(dev)
    model.pooler = model.pooler.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.config.max_position_embeddings, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, *inputs):
            inp = inputs[0]
            inps[cache['i']] = inp
            cache['i'] += 1
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(input_ids=batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.embeddings = model.embeddings.cpu()
    model.pooler = model.pooler.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)


    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['attention.self.query', 'attention.self.key', 'attention.self.value'],
                ['attention.output.dense'],
                ['intermediate.dense'],
                ['output.dense']
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0))[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
                quantizers['bert.encoder.layer.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0))[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers

@torch.no_grad()
def bert_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.config.max_position_embeddings

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.encoder.layer

    model.embeddings = model.embeddings.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.config.max_position_embeddings, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.config.max_position_embeddings):((i + 1) * model.config.max_position_embeddings)].to(dev)
        try:
            model(input_ids=batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.bert.embeddings = model.bert.embeddings.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        
        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        lm_logits = model(input_ids=hidden_states)[0]
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.config.max_position_embeddings):((i + 1) * model.config.max_position_embeddings)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.config.max_position_embeddings
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.config.max_position_embeddings))
    print(ppl.item())

    model.config.use_cache = use_cache


def quantize_bert(model, dev, args):
    print('Quantizing model layers...')

    # Extract encoder layers
    layers = model.encoder.layer
    
    # Move layers to device
    for i in range(len(layers)):
        layers[i] = layers[i].to(dev)
    
    # Perform quantization on each layer
    for i in range(len(layers)):
        print(f'Quantizing layer {i}')
        layer = layers[i]
        
        if args.nearest:
            subset = find_layers(layer)  # 获取层的子集（假设 find_layers 是自定义函数）
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=True
                )
                W = subset[name].weight.data
                if args.groupsize==-1:
                    quantizer.find_params(W, weight=True)
                    subset[name].weight.data = quantize(
                        W, quantizer.scale, quantizer.zero, quantizer.maxq
                    ).to(next(iter(layer.parameters())).dtype)
                else:
                    for j in range(0, W.shape[1], args.groupsize):
                        quantizer.find_params(W[:, j:(j + args.groupsize)], weight=True)
                        subset[name].weight.data[:, j:(j + args.groupsize)] = quantize(
                            W[:, j:(j + args.groupsize)], quantizer.scale, quantizer.zero, quantizer.maxq
                        ).to(next(iter(layer.parameters())).dtype)

        
        # Clean up and move layer back to CPU
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    print('Quantization complete.')
    
    return model

def bert_pack3(model, quantizers):
    layers = find_layers(model)
    print(layers)
    layers = {n: layers[n[5:]] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model

if __name__ == '__main__':
    import argparse
    from GPTQ.datautils import *

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
        '--groupsize', type=int, default=16,
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
        '--save', type=str, default='gptq-xiaobu-2bit-sample1000-group16',
        help='Save quantized checkpoint under this name.'
    )

    args = parser.parse_args()

    device='cuda:0'

    model = get_bert(args.model)
    model.eval()
    print(model)

    if args.nearest:
        model.load_state_dict(torch.load('/home/amax/chx/vsremote/MAB-FG/EmbeddingModels/models/gptq-xiaobu-2bit-sample1000-group16'))

    
    if not args.nearest:
        dataloader, _= get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.config.max_position_embeddings
        )

        if args.wbits < 16:
            tick = time.time()
            quantizers = bert_sequential(model, dataloader, device)
            print(time.time() - tick)
        
        if args.save is not None:
            bert_pack3(model, quantizers)
            torch.save(model.state_dict(), args.save) 
    
    
    model=quantize_bert(model,device,args)


    
    
    from sentence_transformers import SentenceTransformer
    sentence_model=SentenceTransformer(args.model)
    sentence_model._first_module().auto_model=model
    sentence_model.to(device)
    

    task_list = ["JDReview", "MultilingualSentiment", "OnlineShopping", "Waimai",
                "CLSClusteringS2S.v2", "CLSClusteringP2P.v2", "ThuNewsClusteringS2S.v2", "ThuNewsClusteringP2P.v2",
                "Ocnli", "Cmnli",
                "T2Reranking", "MMarcoReranking", "CMedQAv1-reranking", "CMedQAv2-reranking",
                "T2Retrieval", "MMarcoRetrieval", "DuRetrieval", "CovidRetrieval", "CmedqaRetrieval", "EcomRetrieval", "MedicalRetrieval", "VideoRetrieval",
                "ATEC", "BQ", "LCQMC", "PAWSX", "STSB", "AFQMC", "QBQTC"]
    
    #task_list = ["JDReview"]

    for t in task_list:
        tasks = mteb.get_tasks(tasks=[t])
        evaluation = mteb.MTEB(tasks=tasks)
        results = evaluation.run(sentence_model, output_folder=f"results1", encode_kwargs={'batch_size': 256})
    

