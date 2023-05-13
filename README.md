# E6998-HPML-TermProject

## The Secret Sauce: How Post-Training Optimization Can SuperCharge your AI

The overall objective of this project is to optimize inference of pre-trained large visual-language models without significantly sacrificing their performance

Key Objectives 
- Accelerated Inference
- Model Compressions
- Multi-modal Processing

### Data

The model was evaluated on the benchmark datasets ImageNet and CIFAR100. ImageNet is a large-scale image classification dataset with over 1.4 million images and 1000 object categories. CIFAR100 is a subset of the Tiny Images dataset which contains 60,000 32x32 color images and 100 classes.

### Evaluation

In this project we evaluate the model using both task specific and generic metrics namely

- Accuracy
- Compression Ratio
- Inference Time

### Repo Structure

- utils/CLIP_CIFAR_100.py defines the functions for evaluating the model performance on the CIFAR100 dataset
- utils/CLIP_ImageNet.py defines the functions for evaluating the model performance on the ImageNet dataset
- utils/model_compression.py defined the functions to compress the self attention weight matrices in the Transformer architecture
- utils/compression.py implements test runs of compressing the attention blocks for the ImageNet datset
- utils/model_compression_optimum.py defines the functions to perform low rank methods with singular value decomposition and allows the user to specify a compression ratio and which layers to compress
<br>

- data/ Imagenet.json contains the list of 1000 classes of the ImageNet dataset
- CLIP_Inference_CIFAR100.ipynb contains the code to run a sample inference on the CIFAR100 dataset using the CLIP model. The top-1 and top-5 accuracy is measured
<br>

- vision_only_from_start_on_attention.py is used to compress the visual transformer attention blocks (ImageNet dataset)
- vision_only_from_start_on_attention_CIFAR.py is used to compress the visual transformer attention blocks (CIFAR100 dataset)
- vision_only_from_start_IMAGENET_without_attention.py is used to prune the text encoder attention modules (ImageNet dataset)
- Visualize_Result.ipynb contains the code and outputs for generating the graphs of the results of the experiments. This includes compressing the visual transformer attention blocks (constant compression ratio and varying n_layers), compressing the visual transformer attention blocks (constant n_layers and varying compression ratio), and pruning the text encoder attention modules (accuracy/time vs number of layers)
- Visualize_Result.py contains the scipts for generating the graphs of the results of the experiments. 
<br>

- /results/vision_only_from_start_with_attention.json contains the results of compressing the Visual Transformer Attention Blocks for the ImageNet dataset
- /results/vision_only_from_start_with_attention_CIFAR100.json contains the results of compressing the Visual Transformer Attention Blocks for the CIFAR190 dataset
- /results/vision_only_from_start_without_attention.json contains results of pruning the text encoder attention modules

### Example commands to run

- python Visualize_Results.py to visualize the results of the the experiments. Can also observe the output graphs in Visualize_Results.ipynb
- python vision_only_from_start_on_attention_CIFAR.py to run the script to compress the visual transformer attention blocks (CIFAR100 dataset) 
- python vision_only_from_start_on_attention.py to run the script to compress the visual transformer attention blocks (ImageNet dataset)
- python vision_only_from_start_IMAGENET_without_attention.py to run the script to prune the text encoder attention modules (ImageNet dataset)

### Observations

We found out that keeping a low-compression ratio with more number of layers resulted in a better compressed model than keeping a high compression-ratio for a fewer number of layers. We also found out that visual transformer attention layers are less sensitive to low-matrix approximation as compared to their textual counterparts. Therefore we adopted matrix-approximation for the visual transformers only. We experimented with pruning of different attention blocks in the textual encoder of the CLIP. We found that the blocks at the end are less sensitive towards the inference accuracy as compared to the residual attention blocks from the start. We observed that there was a significant dip in the accuracy if we remove more than 3 attention blocks. Overall, this study provides valuable insights for improving the efficiency of CLIP's inference process, making it more viable for real-time and resource-constrained scenarios. 


### References

[1] Hsu, Y.-C. et al. (2022) “Language Model Compression with Weighted Low-Rank Factorization”, International Conference on Learning Representations. Available at: https://doi.org/10.48550/arXiv.2207.00112. 

[2] Tay, Y. et al. (2022) “Efficient transformers: A survey,” ACM Computing Surveys, 55(6), pp. 1–28. Available at: https://doi.org/10.1145/3530811 

[3] Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. arXiv preprint arXiv:2103.00020. https://doi.org/10.48550/arXiv.2103.00020.

[4] Li, J. et al. (2022) "Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation" Available at: https://arxiv.org/pdf/2201.12086.pdf
