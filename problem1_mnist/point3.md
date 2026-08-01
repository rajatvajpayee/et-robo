| Condition                    | Verification                    | Remarks                                                                                                                                                                                                                                                  |
| ---------------------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Ergodicity of input data** | **Reasonably satisfied**        | MNIST samples are randomly shuffled every epoch (`shuffle=True`). Each mini-batch is an approximately i.i.d. sample from the training distribution, satisfying the practical assumption used by SGD.                                                     |
| **Step size conditions**     | **Partially satisfied**         | A fixed learning rate (e.g., 0.01) was used. This satisfies stability for training but does **not** satisfy the theoretical Robbins-Monro conditions (`∑η_t = ∞`, `∑η_t² < ∞`) required for convergence proofs because the learning rate is not decayed. |
| **Unbiased gradient**        | **Approximately satisfied**     | Mini-batches are sampled uniformly at random, so the stochastic gradient is an unbiased estimate of the full-batch gradient in expectation. No importance sampling or biased sampling is introduced.                                                     |
| **Bounded variance**         | **Cannot be formally verified** | The variance of stochastic gradients was not mathematically bounded. Empirically, training remained stable with no exploding gradients or divergence, suggesting finite gradient variance.                                                               |
| **L-smoothness**             | **Cannot be verified**          | Proving Lipschitz smoothness requires theoretical analysis of the loss surface or Hessian bounds. For CNNs with ReLU activations, this property is generally assumed locally but was not proven in this implementation.                                  |



These conditions come from the **theory of Stochastic Gradient Descent (SGD)**. They are assumptions under which mathematicians can prove that SGD converges to a local optimum (or stationary point for non-convex problems like CNNs). Let's go through them one by one.

---

# 1. Ergodicity of Input Data

### What is it?

Ergodicity means that **the mini-batches you see during training are representative of the overall data distribution**.

Suppose your dataset has digits:

```
0 1 2 3 4 5 6 7 8 9
```

If your batches are

```
Batch 1 : only 0s
Batch 2 : only 1s
Batch 3 : only 2s
...
```

then SGD receives biased information.

Instead, after shuffling, a batch looks like

```
Batch:
3 7 1 5 8 0 9 2 ...
```

Each mini-batch approximately follows the whole dataset distribution.

### Why is it important?

SGD assumes each batch provides information about the full dataset.

Otherwise, gradients become biased.

### In your project

You use

```python
DataLoader(
    train_dataset,
    shuffle=True
)
```

Therefore

> Every epoch randomly shuffles MNIST, making each mini-batch approximately i.i.d. and representative of the full data distribution.

Hence ergodicity is approximately satisfied.

---

# 2. Step Size Conditions

Step size means

```
Learning Rate
```

Suppose

```
θ = θ - η∇L
```

Here

```
η = learning rate
```

---

### Why is it important?

If η is too large

```
Loss

\
 \
  \
   \__
      \
       /
      /
```

The optimizer jumps around and never converges.

If η is too small

```
takes forever
```

---

### Theoretical requirement

SGD convergence proofs require

[
\sum_{t=1}^{\infty}\eta_t=\infty
]

meaning

learning never completely stops.

Also

[
\sum_{t=1}^{\infty}\eta_t^2<\infty
]

meaning

learning rate eventually becomes sufficiently small.

A common schedule is

```
ηt = 1/t
```

which satisfies both.

---

### Your implementation

You use

```python
lr = 0.01
```

throughout training.

Therefore

The theoretical condition is **not satisfied** because the learning rate never decreases.

However,

for practical deep learning,

constant learning rates are extremely common.

---

# 3. Unbiased Gradient

This is probably the most important assumption.

Suppose the dataset has

```
60,000 images
```

The true gradient is

```
gradient over all 60,000 images
```

But we compute

```
gradient over only 64 images
```

Why is that okay?

Because if batches are sampled randomly,

the expected gradient equals the full gradient.

Mathematically,

[
E[g_t]=\nabla L(\theta)
]

meaning

the average stochastic gradient equals the true gradient.

---

### Example

Suppose full gradient is

```
10
```

Mini-batch gradients

```
8
11
12
9
10
```

Average

```
10
```

Exactly the full gradient.

That's unbiased.

---

### In your implementation

Since batches are randomly sampled,

your gradient estimate is approximately unbiased.

---

# 4. Bounded Variance

Although SGD is unbiased,

individual batches are noisy.

Example

```
Batch 1 gradient = 5

Batch 2 gradient = 20

Batch 3 gradient = 13

Batch 4 gradient = 7
```

There is variance.

Theory assumes

the variance is finite.

Mathematically,

[
E|g-\nabla L|^2\le\sigma^2
]

meaning

gradient noise cannot become infinitely large.

---

### Why?

If gradients vary too much,

training becomes unstable.

```
Loss

\
 \
 /
 \
 /
 \
```

oscillates forever.

---

### Can you verify it?

No.

You would need

```
Compute gradient
Compute variance
Repeat over many batches
```

Even then,

you cannot prove it theoretically.

Therefore

> Cannot be formally verified.

---

# 5. L-smoothness

This is the hardest one.

A function is L-smooth if

its gradient does not change abruptly.

Instead of

```
\
 \
 |
 |
____
```

which has sharp changes,

the loss should look more like

```
\
 \
  \
   \
```

smoothly changing.

---

Mathematically,

[
|\nabla f(x)-\nabla f(y)|
\le
L|x-y|
]

meaning

small parameter changes produce small gradient changes.

---

### Why?

If gradients suddenly become enormous,

optimization becomes unstable.

L-smoothness guarantees

```
small step

↓

small gradient change
```

---

### Can you verify this?

Practically,

No.

You would need

* Hessian matrix
* Spectral norm
* Lipschitz constant

for every point in parameter space.

Even research papers usually assume this instead of proving it.

---

# Summary

| Condition             | Meaning                                          | Your Project                                 |
| --------------------- | ------------------------------------------------ | -------------------------------------------- |
| **Ergodicity**        | Mini-batches represent the full dataset          | ✅ Yes (`shuffle=True`)                       |
| **Step size**         | Learning rate should eventually decrease         | ⚠️ Partially (constant LR)                   |
| **Unbiased Gradient** | Average mini-batch gradient equals full gradient | ✅ Approximately yes                          |
| **Bounded Variance**  | Gradient noise should remain finite              | ❌ Cannot prove, only observe stable training |
| **L-smoothness**      | Loss surface should change smoothly              | ❌ Cannot verify theoretically                |

For an implementation assignment, this level of explanation is typically what instructors expect: identify each assumption, explain why it matters, state whether your implementation satisfies it, and justify any assumptions that cannot be verified experimentally.
