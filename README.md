# Transformer From First Principles

A small encoder-only Transformer implemented from first principles, starting from the mathematical foundations of attention and gradually building the complete architecture in PyTorch.

The purpose of this project is not to build a competitive language model. It is to understand what actually happens inside a Transformer by implementing its important components ourselves rather than relying on ready-made Transformer modules.

The experiment uses a completely synthetic dataset, allowing the model architecture to remain the main focus.

---

## 1. Why Build a Transformer From First Principles?

Modern NLP systems often make the Transformer appear deceptively simple:

```text
Text
  ↓
Tokenizer
  ↓
Transformer
  ↓
Prediction
```

In reality, the Transformer is a sequence of mathematical operations.

A token is converted into a vector. Position information is added. Those vectors interact through attention. The resulting representation passes through normalization and a feed-forward network. These operations are repeated across encoder layers before the final representation is used for prediction.

The goal of this project is to expose those operations one by one.

Instead of starting with:

```python
TransformerEncoder(...)
```

the project builds the important pieces explicitly:

```text
Tokenization
    ↓
Embedding
    ↓
Positional Encoding
    ↓
Query / Key / Value projections
    ↓
Scaled Dot-Product Attention
    ↓
Multi-Head Attention
    ↓
Residual Connection
    ↓
Layer Normalization
    ↓
Feed-Forward Network
    ↓
Residual Connection
    ↓
Layer Normalization
    ↓
Encoder Stack
    ↓
[CLS] Representation
    ↓
Classification
```

---

## 2. The Task

Before building the model, we need a problem simple enough that we can concentrate on understanding the architecture.

The model receives a small collection of synthetic facts followed by a query.

For example:

```text
Alice likes Bob.
Bob follows Charlie.
Charlie knows David.

Query Alice likes Bob?
```

The expected answer is:

```text
1
```

Another example might be:

```text
Alice knows David.
David follows Bob.

Query Ian likes Jack?
```

The answer is:

```text
0
```

The dataset is generated programmatically, so there is no external corpus or pretrained knowledge involved.

The task is intentionally simple. The interesting part is not the dataset itself; it is understanding how the Transformer converts this sequence into a useful representation.

---

## 3. Synthetic Language

The vocabulary contains a small set of names, relations, punctuation tokens, and control tokens.

Conceptually, the input looks like:

```text
[CLS]
Alice
knows
David
.
David
follows
Bob
.
Query
Ian
likes
Jack
?
[SEP]
```

The model therefore receives a sequence rather than individual words independently.

This is important because self-attention operates across the entire sequence.

---

## 4. Tokenization From Scratch

The first step is converting text into discrete token IDs.

A small vocabulary is constructed from the synthetic training corpus.

Special tokens are reserved for structural purposes:

```text
[PAD]
[UNK]
[CLS]
[SEP]
```

For example:

```text
Alice → 6
knows → 19
David → 9
.     → 4
```

The sentence therefore becomes a sequence of integers.

The actual experiment produced a vocabulary of:

```text
22 tokens
```

The encoded sequence starts with `[CLS]` and ends with `[SEP]`.

Padding is then added until every sequence has the same length.

---

## 5. From Tokens to Vectors

A neural network cannot directly reason with token IDs.

The integer:

```text
Alice → 6
```

does not contain useful mathematical meaning by itself.

We therefore create a trainable embedding matrix:

\[
E \in \mathbb{R}^{V \times d}
\]

where:

- \(V\) = vocabulary size
- \(d\) = embedding dimension

For this experiment:

\[
V = 22
\]

and:

\[
d = 32
\]

Therefore:

\[
E \in \mathbb{R}^{22 \times 32}
\]

Each token selects one row from this matrix.

So:

```text
Token ID
   ↓
Embedding lookup
   ↓
32-dimensional vector
```

The model can now learn useful representations for tokens during training.

---

## 6. The Problem With Embeddings

There is an important problem.

Consider:

```text
Alice likes Bob
```

and:

```text
Bob likes Alice
```

The same tokens are present, but their order is different.

A normal embedding lookup does not know whether a token appeared at position 1 or position 3.

Transformers therefore need explicit positional information.

---

## 7. Sinusoidal Positional Encoding

This project uses the original sinusoidal positional encoding idea.

For position \(pos\) and embedding dimension \(i\):

\[
PE(pos,2i)
=
\sin
\left(
\frac{pos}
{10000^{2i/d}}
\right)
\]

and:

\[
PE(pos,2i+1)
=
\cos
\left(
\frac{pos}
{10000^{2i/d}}
\right)
\]

The positional vector is added directly to the token embedding:

\[
X = E + PE
\]

This gives every token information about both:

```text
What am I?
```

and:

```text
Where am I?
```

The experiment uses:

\[
40 \times 32
\]

positional values because the maximum sequence length is 40.

---

## 8. Self-Attention

This is the central idea behind the Transformer.

Instead of processing every token independently, self-attention allows every token to interact with other tokens in the sequence.

For an input matrix \(X\), three different representations are created:

\[
Q = XW_Q
\]

\[
K = XW_K
\]

\[
V = XW_V
\]

These are called:

```text
Q → Query
K → Key
V → Value
```

The intuition is:

- Query asks what information this token is looking for.
- Key describes what each token contains.
- Value contains the information that can actually be passed forward.

---

## 9. Attention Scores

Queries and keys are compared using a dot product:

\[
S = QK^T
\]

A large value means that a query and key are strongly aligned.

However, the dot products can become large when the dimensionality increases.

Therefore the scores are scaled:

\[
S =
\frac{QK^T}
{\sqrt{d_k}}
\]

where \(d_k\) is the dimension of each attention head.

For this model:

\[
d_k =
\frac{32}{4}
=
8
\]

---

## 10. Turning Scores Into Probabilities

The scaled scores are passed through softmax:

\[
A =
softmax
\left(
\frac{QK^T}
{\sqrt{d_k}}
\right)
\]

This produces attention weights.

For a particular token, the weights determine how much information it should take from every other token.

Conceptually:

```text
Query
  │
  ├── Token 1 → 0.05
  ├── Token 2 → 0.10
  ├── Token 3 → 0.65
  ├── Token 4 → 0.15
  └── Token 5 → 0.05
```

The weights sum to approximately 1.

---

## 11. Applying Attention

The attention weights are multiplied by the value vectors:

\[
Z = AV
\]

Combining the previous equations gives the complete scaled dot-product attention equation:

\[
\boxed{
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}
{\sqrt{d_k}}
\right)V
}
\]

This is the mathematical core of self-attention.

---

## 12. Why Multiple Heads?

One attention operation has only one learned projection space.

Transformers therefore use multiple attention heads.

Instead of:

```text
Input
  ↓
One attention operation
```

we use:

```text
Input
  ↓
 ┌───────┬───────┬───────┬───────┐
 │ Head 1│ Head 2│ Head 3│ Head 4│
 └───────┴───────┴───────┴───────┘
```

Each head learns its own:

\[
W_Q,\ W_K,\ W_V
\]

and can therefore learn different interaction patterns.

For this experiment:

\[
d_{model}=32
\]

and:

\[
h=4
\]

so each head receives:

\[
d_k = \frac{32}{4}=8
\]

dimensions.

The outputs of the heads are concatenated:

\[
Z =
Concat(Z_1,Z_2,\ldots,Z_h)
\]

and projected again:

\[
Output = ZW_O
\]

---

## 13. The Transformer Encoder Layer

Attention alone is not the complete Transformer.

One encoder layer contains two major sub-blocks:

```text
Input
  │
  ▼
Multi-Head Self-Attention
  │
  ▼
Residual Connection
  │
  ▼
Layer Normalization
  │
  ▼
Feed-Forward Network
  │
  ▼
Residual Connection
  │
  ▼
Layer Normalization
  │
  ▼
Output
```

This project uses two such encoder layers.

---

## 14. Residual Connections

Instead of completely replacing the input with the output of a sub-layer, the original representation is added back:

\[
Y = X + Attention(X)
\]

and later:

\[
Z = Y + FFN(Y)
\]

These residual paths allow information to flow directly through the network and make deeper architectures easier to optimize.

---

## 15. Layer Normalization

After each residual connection, the representation is normalized.

For an input vector \(x\):

\[
\mu =
\frac{1}{d}
\sum_{i=1}^{d}x_i
\]

and:

\[
\sigma^2 =
\frac{1}{d}
\sum_{i=1}^{d}(x_i-\mu)^2
\]

The normalized representation is:

\[
\hat{x}
=
\frac{x-\mu}
{\sqrt{\sigma^2+\epsilon}}
\]

Trainable scale and shift parameters then produce:

\[
y =
\gamma\hat{x}+\beta
\]

The implementation manually calculates these statistics and applies trainable \(\gamma\) and \(\beta\).

---

## 16. Feed-Forward Network

After attention, each token representation independently passes through a small neural network:

\[
FFN(x)
=
W_2
\,
ReLU(W_1x+b_1)
+b_2
\]

The dimensionality used here is:

```text
32
 ↓
64
 ↓
32
```

The attention mechanism mixes information between tokens.

The feed-forward network then transforms each token representation independently.

This separation is an important part of the Transformer design.

---

## 17. Complete Encoder

Putting everything together:

```text
Input Tokens
     │
     ▼
Token Embedding
     │
     +
     │
Positional Encoding
     │
     ▼
┌───────────────────────────────┐
│ Transformer Encoder Layer 1   │
│                               │
│ Multi-Head Self-Attention     │
│          ↓                    │
│ Residual + LayerNorm          │
│          ↓                    │
│ Feed-Forward Network          │
│          ↓                    │
│ Residual + LayerNorm          │
└───────────────┬───────────────┘
                │
                ▼
┌───────────────────────────────┐
│ Transformer Encoder Layer 2   │
│                               │
│ Multi-Head Self-Attention     │
│          ↓                    │
│ Residual + LayerNorm          │
│          ↓                    │
│ Feed-Forward Network          │
│          ↓                    │
│ Residual + LayerNorm          │
└───────────────┬───────────────┘
                │
                ▼
           [CLS] Vector
                │
                ▼
        Linear Classifier
                │
                ▼
             YES / NO
```

The final model is therefore **encoder-only**.

There is no decoder and no autoregressive text generation.

---

## 18. Why `[CLS]`?

The Transformer produces one representation for every position:

\[
H \in \mathbb{R}^{L \times d}
\]

where \(L\) is the sequence length.

The special `[CLS]` token is placed at the beginning of the sequence.

Its final representation is used as a summary representation:

\[
h_{CLS}=H_0
\]

The classification head then computes:

\[
logits = W_ch_{CLS}+b_c
\]

For this binary task:

\[
32 \rightarrow 2
\]

The two outputs correspond to the two possible classes.

---

## 19. Training Objective

The model produces two logits:

\[
z_0,\ z_1
\]

Softmax converts them into class probabilities:

\[
P(y=i)
=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
\]

Training minimizes cross-entropy:

\[
\mathcal{L}
=
-\log P(y_{true})
\]

The parameters are updated using AdamW.

The important point is that the gradients flow through the complete architecture:

```text
Classifier
    ↓
[CLS]
    ↓
Encoder
    ↓
Attention
    ↓
Q/K/V projections
    ↓
Embeddings
```

Therefore the model learns both the final classifier and the internal representations.

---

## 20. Model Configuration

The experiment intentionally keeps the model small.

| Component | Configuration |
|---|---:|
| Vocabulary | 22 tokens |
| Maximum sequence length | 40 |
| Embedding dimension | 32 |
| Encoder layers | 2 |
| Attention heads | 4 |
| Dimension per head | 8 |
| Feed-forward dimension | 64 |
| Output classes | 2 |
| Trainable parameters | 17,858 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Optimizer | AdamW |

The resulting model contains only:

\[
17,858
\]

trainable parameters.

This is deliberately tiny compared with modern language models.

---

## 21. Implementation Philosophy

The project avoids high-level Transformer implementations.

The following components are implemented directly:

```text
Tokenizer
Embedding lookup
Sinusoidal positional encoding
Q/K/V projections
Scaled dot-product attention
Multi-head attention
Residual connections
Layer normalization
Feed-forward network
Transformer encoder
Classification head
Training loop
Evaluation
Synthetic dataset generation
```

PyTorch is used primarily as the numerical and automatic-differentiation backend.

The purpose is to understand the architecture rather than recreate an entire deep-learning framework.

---

## 22. Project Structure

```text
transformer-from-first-principles/
│
├── data/
│   ├── __init__.py
│   └── synthetic_dataset.py
│
├── src/
│   ├── __init__.py
│   ├── attention.py
│   ├── layers.py
│   ├── metrics.py
│   ├── model.py
│   ├── positional_encoding.py
│   ├── tokenizer.py
│   ├── transformer.py
│   └── utils.py
│
├── checkpoints/
│
├── train.py
├── evaluate.py
├── requirements.txt
├── .gitignore
└── README.md
```

The separation is intentional: each major mathematical component has its own implementation.

---

## 23. Experiment Flow

The complete experiment follows:

```text
Synthetic Data
      ↓
Vocabulary Construction
      ↓
Tokenization
      ↓
Padding
      ↓
Embedding
      ↓
Positional Encoding
      ↓
Transformer Encoder
      ↓
[CLS] Representation
      ↓
Classification
      ↓
Cross-Entropy Loss
      ↓
AdamW
      ↓
Validation
      ↓
Best Checkpoint
      ↓
Test Evaluation
```

The training script also prints intermediate information so that the construction of the model can be followed directly from the terminal.

For example, the experiment reports:

```text
Embedding matrix: [22, 32]

Positional encoding: [40, 32]

Transformer layers: 2

Attention:
    d_model  = 32
    heads    = 4
    head_dim = 8

Feed Forward:
    32 → 64 → 32

Classifier:
    32 → 2

Trainable parameters:
    17,858
```

This makes the relationship between the configuration and the mathematical architecture visible during execution.

---

## 24. Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

These metrics provide different views of classification performance rather than relying only on accuracy.

The test set is generated separately from the training set, so the model must process unseen synthetic examples.

---

## 25. What This Project Demonstrates

The most important result of this project is not the classification score.

It is the connection between the equations and the implementation.

For example:

```text
Q = XWQ
```

becomes an actual tensor operation.

```text
QKᵀ / √dk
```

becomes the attention-score calculation.

```text
softmax(...)
```

becomes the attention distribution.

```text
Attention(Q,K,V)
```

becomes the context representation.

And the sequence of these operations becomes an actual Transformer encoder.

This is the main idea behind building the project from first principles: every major block should have a mathematical explanation and a corresponding implementation.

---

## 26. Running the Project

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Generate the synthetic dataset:

```bash
python -m data.synthetic_dataset
```

Train the model:

```bash
python train.py
```

Inspect individual predictions:

```bash
python evaluate.py
```

The best model checkpoint is stored under:

```text
checkpoints/best_model.pt
```

---

## 27. What Comes Next?

This implementation intentionally stops at the encoder.

The next logical step is to make the synthetic task more demanding rather than immediately increasing the model size.

The progression can be:

```text
Exact relationship classification
          ↓
Multi-hop reasoning
          ↓
Negation
          ↓
Variable-length reasoning
          ↓
Masked-token prediction
          ↓
Encoder pretraining
          ↓
Causal self-attention
          ↓
Decoder-only Transformer
          ↓
Next-token prediction
          ↓
Tiny GPT-style model
```

This progression makes the architectural differences easier to understand.

An encoder learns contextual representations from the complete input sequence.

A decoder-only Transformer introduces causal masking so that a token can only attend to earlier positions.

That single change leads toward the architecture used by GPT-style models.

---

## 28. Final Perspective

A Transformer can initially look like a very complicated architecture.

At its core, however, the encoder repeatedly performs a relatively small collection of operations:

\[
\boxed{
Embedding
+
Position
\rightarrow
Attention
\rightarrow
Normalization
\rightarrow
Feed\ Forward
\rightarrow
Normalization
}
\]

The power comes from repeating these operations while allowing every token to dynamically interact with the others through attention.

This project starts with a tiny 22-token vocabulary and a 17,858-parameter model, but the underlying ideas scale to the much larger Transformer architectures used in modern NLP and multimodal systems.

The objective is therefore not to build a large model.

It is to understand the small one well enough that the large ones stop looking mysterious.
