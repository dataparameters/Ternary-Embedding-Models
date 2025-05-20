# Efficient Ternary Weight Embedding Model: Bridging Scalability and Performance

## Overview

This repository provides the implementation for the paper **"Efficient Ternary Weight Embedding Model: Bridging Scalability and Performance"**. The repository includes a novel fine-tuning framework for embedding models to leverage ternary-weight networks. This approach reduces memory and computational overhead while maintaining high performance across natural language processing (NLP) and computer vision (CV) tasks.

Key contributions:
1. **Ternary Quantization**: Introduced ternary-weight embedding models (-1, 0, +1) for better efficiency while maintaining comparable accuracy.
2. **Self-Distillation**: A fine-tuning framework leveraging outputs from pre-trained full-precision models to guide ternary-weight networks.
3. **Scalable Performance**: Demonstrated improvements in memory efficiency and latency in text and image embedding models, with integration in Approximate Nearest Neighbor (ANN) search for real-time recommendation systems.

## Requirements

- Python >= 3.8
- torch
- transformers
- sentence-transformers
- faiss
- numpy
- tqdm
- datasets
- torchvision (for image tasks)
- mteb
- (bitblas, if used)

Please install these dependencies manually as there is no `requirements.txt` file in the repository.

## Directory Structure

- `text_embedding/`: Scripts for training and evaluating ternary text embedding models.
- `image_embedding/`: Scripts for training and evaluating ternary image embedding models.
- `Embedding-ANN/`: Scripts for evaluating embeddings with ANN search.
- `GPTQ-Bert/`: Scripts for quantized BERT models (if relevant).

## Usage

### Text Embedding

- **Training and Evaluation**:
  ```bash
  python text_embedding/train_and_eval_TnModel.py
  ```
  Edit the script to set your model path, device, and options for ternary/bitblas as needed.

- **Benchmarking Speed/Memory**:
  ```bash
  python text_embedding/bitblas_speed_memory.py
  ```

### Image Embedding

- **Training and Evaluation**:
  ```bash
  python image_embedding/train_classifier_and_eval.py
  ```
  Edit the script to set dataset/model paths and options.

- **Benchmarking Speed/Memory**:
  ```bash
  python image_embedding/bitblas_speed_memory.py
  ```

- **BitBlas Accuracy Evaluation**:
  ```bash
  python image_embedding/bitblas_accuracy.py
  ```

### ANN Evaluation

- **Run ANN Evaluation**:
  ```bash
  python Embedding-ANN/ANN_eval.py
  ```
  Edit the script to set model paths and ANN method.

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
