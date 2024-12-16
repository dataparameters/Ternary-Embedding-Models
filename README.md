# Efficient Ternary Weight Embedding Model: Bridging Scalability and Performance

## Overview

This repository provides the implementation for the paper **"Efficient Ternary Weight Embedding Model: Bridging Scalability and Performance"**. The repository includes a novel fine-tuning framework for embedding models to leverage ternary-weight networks. This approach reduces memory and computational overhead while maintaining high performance across natural language processing (NLP) and computer vision (CV) tasks.

Key contributions:
1. **Ternary Quantization**: Introduced ternary-weight embedding models (-1, 0, +1) for better efficiency while maintaining comparable accuracy.
2. **Self-Distillation**: A fine-tuning framework leveraging outputs from pre-trained full-precision models to guide ternary-weight networks.
3. **Scalable Performance**: Demonstrated improvements in memory efficiency and latency in text and image embedding models, with integration in Approximate Nearest Neighbor (ANN) search for real-time recommendation systems.

## Requirements

- Python >= 3.8
- PyTorch >= 1.10
- Hugging Face Transformers
- FAISS
- NumPy, Pandas, Matplotlib
- Other dependencies in `requirements.txt`

To install the dependencies, run:

```bash
pip install -r requirements.txt
```

## Getting Started

### Fine-Tuning Ternary-Weight Models

#### Text Embedding
1. **Prepare Dataset**: Place the training data (e.g., `nli-zh-25k` and `t2ranking`) in the `data/` folder.
2. **Run Fine-Tuning**:
   ```bash
   python experiments/text_embedding/fine_tune_text.py --model xiaobu-embedding-v2 --data_path data/
   ```

#### Image Embedding
1. **Prepare Dataset**: Use datasets like ImageNet-1k or CIFAR for training.
2. **Run Fine-Tuning**:
   ```bash
   python experiments/image_embedding/fine_tune_image.py --model vit-base-patch16-224 --data_path data/
   ```

### Evaluation
1. **Evaluate Text Embedding**:
   ```bash
   python experiments/text_embedding/evaluate_text.py --model xiaobu-ternary
   ```
2. **Evaluate Image Embedding**:
   ```bash
   python experiments/image_embedding/evaluate_image.py --model vit-ternary
   ```

### Integration with ANN
1. **Run ANN Integration**:
   ```bash
   python experiments/ann/ann_integration.py --model xiaobu-ternary --dataset CmedqaRetrieval
   ```

2. **Evaluate ANN Performance**:
   ```bash
   python experiments/ann/ann_evaluation.py --model xiaobu-ternary --dataset CmedqaRetrieval
   ```

## Results

### Text Embedding
| **Task**          | **Baseline (FP32)** | **Ternary (INT2)** | **Latency** | **Storage** |
|--------------------|---------------------|--------------------|-------------|-------------|
| Retrieval          | 82.27              | 78.38             | 0.37×       | 0.13×       |
| Classification     | 72.70              | 70.14             | 0.37×       | 0.13×       |
| STS                | 64.18              | 62.70             | 0.37×       | 0.13×       |

### Image Embedding
| **Dataset**       | **Baseline (FP32)** | **Ternary (INT2)** | **Latency** | **Storage** |
|--------------------|---------------------|--------------------|-------------|-------------|
| CIFAR-10          | 96.79%             | 94.79%            | 0.69×       | 0.07×       |
| CIFAR-100         | 86.38%             | 82.36%            | 0.69×       | 0.07×       |

## Citation

If you use this code or framework in your research, please cite:

```bibtex
@article{chen2024ternaryembedding,
  title={Efficient Ternary Weight Embedding Model: Bridging Scalability and Performance},
  author={Chen, Jiayi and Wu, Chen and Zhang, Shaoqun and Li, Nan and Zhang, Liangjie and Zhang, Qi},
  journal={arXiv preprint arXiv:2411.15438},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the teams behind [LAMDA](https://www.lamda.nju.edu.cn/MainPage.ashx), and the open datasets used in this project.
