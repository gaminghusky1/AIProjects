# AIProjects

Welcome to AIProjects, a collection of machine learning projects exploring neural networks, deep learning, and large
language models, as well as various other experiments related to AI.

This is an ongoing project, and I often add new projects and experiments exploring different areas of machine learning,
such as new layers, training on a new dataset, etc.

# Projects
## 1. Neural Networks From Scratch

The files for this project are contained within the `Neural Networks From Scratch` folder.

This is the biggest project in this repository, where I built (and am still in the process of improving and adding to) 
a modular neural network framework implemented in Python, using two similar libraries:

- **NumPy:** The files for this implementation are contained within the `base_model` package, and it includes various
layers, loss functions, activations, and optimizers, with a `model` class to tie everything together. Its main focus is
on forward/backpropogation and gradient descent across multiple layers. This is the original module that I implemented
before shifting my focus to MLX.
- **Apple MLX:** The files for the MLX implementation are contained within the `mlx_model` package. It is a GPU-accelerated
version of the NumPy module that leverages Apple's MLX framework to accelerate training on Apple Silicon.
This is the most up-to-date model with the most layers and components.

**Some of the components I have implemented so far:**
- Dense and Convolutional layers
- Max pooling, dropout, flattening, and reshaping
- Various activation and loss functions, including softmax and categorical cross-entropy
- Adam and AdamW optimizers for gradient descent
- Batch training and learning rate scheduling
- Training metric (Loss, Accuracy, EMA Loss, etc.) tracking

There are also many other layers that I have implemented within this module, but I will go into more detail on them
in the next section on Transformers.

**Experiments and applications I have used this module for:**
- Handwritten digit classification with the MNIST dataset
- Image classification with CIFAR-10 and CIFAR-100 datasets
- Interfaces to visualize model performance and predictions

## 2. Transformers

This is a subproject of `Neural Networks from Scratch`, focused on building transformer language models. It primarily
uses the MLX module in `mlx_model` for faster training of the large models. 

**Specific layers and components used for Transformers:**
- Multi-headed self-attention and causal masking
- Rotary Positional Embeddings (RoPE)
- Token Embeddings and weight tying
- RMSNorm and LayerNorm
- Residual Blocks
- KV Cache for faster inference

**Transformer Projects:**
- **TinyChat:** The first transformer language model I made using my framework, trained on conversational data generated
by Gemini in the TinyChat dataset. It has 24M parameters and a context/sequence length of 256 tokens.
- **SmolLM/SmolTalk:** A larger project pretrained on SmolLM for knowledge and fine-tuned on SmolTalk for a conversational
tone. It has 54M parameters and a context/sequence length of 1024 tokens.

**TinyChat train metrics:**

![](Metric%20Images/tinychat_train_loss.png)
![](Metric%20Images/tinychat_train_ema_loss.png)
![](Metric%20Images/tinychat_train_accuracy.png)

**SmolLm/SmolTalk train metrics:**

![](Metric%20Images/smollm_smoltalk_train_loss.png)
![](Metric%20Images/smollm_smoltalk_train_ema_loss.png)
![](Metric%20Images/smollm_smoltalk_train_accuracy.png)

## 3. Additional Experiments

- **Regression:** A simple implementation of linear regression using orthogonal projections and other concepts from linear algebra
- **Overfitting Test:** An experiment investigating overfitting on CIFAR-10 with Dense and Convolutional models that I
conducted for AP Statistics