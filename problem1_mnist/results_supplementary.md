# Model Evaluation Results

The final model was trained using the hyperparameters selected from the convergence analysis.

| Hyperparameter | Value |
|---------------|------:|
| Optimizer | SGD |
| Learning Rate | 0.01 |
| Batch Size | 256 |
| Epochs | 20 |

---

## Model Architectures

### Baseline

```yaml
input_channels: 1
num_classes: 10

cnn:
  filters: [8]
  kernel_size: 3
  pool_size: 2

mlp:
  hidden_dims: [32]

activation: relu
```

---

### Model 1

```yaml
input_channels: 1
num_classes: 10

cnn:
  filters: [16]
  kernel_size: 3
  pool_size: 2

mlp:
  hidden_dims: [64]

activation: relu
```

---

### Model 2

```yaml
input_channels: 1
num_classes: 10

cnn:
  filters: [16]
  kernel_size: 3
  pool_size: 2

mlp:
  hidden_dims: [128]

activation: relu
```

---

### Model 3

```yaml
input_channels: 1
num_classes: 10

cnn:
  filters: [16, 32]
  kernel_size: 3
  pool_size: 2

mlp:
  hidden_dims: [128]

activation: relu
```

---

### Model 4

```yaml
input_channels: 1
num_classes: 10

cnn:
  filters: [16, 32]
  kernel_size: 3
  pool_size: 2

mlp:
  hidden_dims: [256]

activation: relu
```

---

# Quantitative Results

| Model | Filters | Hidden MLP | ROC-AUC | Precision | Recall | F1 | Accuracy |
|:------|:-------:|:----------:|---------:|----------:|--------:|---:|----------:|
| Baseline | [8] | 32 | 0.9991 | 0.9690 | 0.9689 | 0.9689 | 0.9689 |
| Model 1 | [16] | 64 | 0.9993 | 0.9711 | 0.9710 | 0.9710 | 0.9710 |
| Model 2 | [16] | 128 | 0.9993 | 0.9693 | 0.9691 | 0.9691 | 0.9691 |
| Model 3 | [16,32] | 128 | 0.9998 | 0.9792 | 0.9790 | 0.9790 | 0.9790 |
| **Model 4** | **[16,32]** | **256** | **0.9998** | **0.9818** | **0.9817** | **0.9817** | **0.9817** |

---

- Increasing the network capacity consistently improved performance.
- Increasing the number of convolutional filters provided a larger performance gain than only increasing the MLP hidden dimension.
- The best performance was achieved by **Model 4**, which consists of two convolutional layers with **16** and **32** filters followed by a **256-neuron MLP**, achieving **98.17%** accuracy and **0.9998 ROC-AUC**.
- Precision, Recall and F1-score are nearly identical for all models, indicating balanced classification performance across classes.

---

# Supplementary Results

The following section contains the confusion matrices for all evaluated models.

## Baseline

### Evaluation Metrics

| Metric | Value |
|--------|------:|
| Loss | 0.1091 |
| Accuracy | 0.9689 |
| Precision | 0.9690 |
| Recall | 0.9689 |
| F1 Score | 0.9689 |
| ROC-AUC | 0.9991 |

### Confusion Matrix

```

Confusion Matrix
         0     1     2     3     4     5     6     7     8     9
 0 |   967     0     1     2     0     3     4     2     1     0
 1 |     0  1123     3     3     0     1     3     1     1     0
 2 |     4     4  1002     6     4     0     3     4     4     1
 3 |     0     0     7   987     0     1     0     3     8     4
 4 |     1     0     5     0   954     0     6     1     2    13
 5 |     4     2     2    11     1   849     6     0     9     8
 6 |     8     2     1     1     5     3   933     1     4     0
 7 |     0     8    22     6     2     0     0   975     2    13
 8 |     6     1     6    10     3     3     1     2   938     4
 9 |     4     6     1    10    15     1     1     7     3   961

```

---

## Model 1

### Evaluation Metrics

| Metric | Value |
|--------|------:|
| Loss | 0.0926 |
| Accuracy | 0.9710 |
| Precision | 0.9711 |
| Recall | 0.9710 |
| F1 Score | 0.9710 |
| ROC-AUC | 0.9993 |

### Confusion Matrix

```
Confusion Matrix
         0     1     2     3     4     5     6     7     8     9
 0 |   968     0     1     0     0     3     4     2     2     0
 1 |     0  1120     3     1     0     0     5     2     4     0
 2 |     6     4   999     3     6     0     3     6     5     0
 3 |     0     0     5   989     0     2     0     3     8     3
 4 |     1     0     3     0   961     0     5     3     2     7
 5 |     7     1     0    11     0   858     7     1     6     1
 6 |     6     2     1     0     5     6   934     1     3     0
 7 |     1     3    16     3     0     0     0   999     3     3
 8 |     5     0     2     9     3     5     6    11   931     2
 9 |     9     6     1     9    18     2     0    12     1   951
```

---

## Model 2

### Evaluation Metrics

| Metric | Value |
|--------|------:|
| Loss | 0.0972 |
| Accuracy | 0.9691 |
| Precision | 0.9693 |
| Recall | 0.9691 |
| F1 Score | 0.9691 |
| ROC-AUC | 0.9993 |

### Confusion Matrix

```
Confusion Matrix
         0     1     2     3     4     5     6     7     8     9
 0 |   963     0     0     1     0     1     7     2     5     1
 1 |     0  1121     2     1     0     2     5     2     2     0
 2 |     8     5   995     3     2     0     2     8     9     0
 3 |     2     0     5   969     0     9     0     4    17     4
 4 |     0     0     4     0   963     0     5     2     2     6
 5 |     6     1     0     7     1   861     7     1     7     1
 6 |     9     3     0     0     7     4   933     0     2     0
 7 |     2     5    12     2     1     0     0   998     4     4
 8 |     5     0     4     4     4     3     5     6   940     3
 9 |     8     6     1     5    24     3     1     6     7   948
```

---

## Model 3

### Evaluation Metrics

| Metric | Value |
|--------|------:|
| Loss | 0.0649 |
| Accuracy | 0.9790 |
| Precision | 0.9792 |
| Recall | 0.9790 |
| F1 Score | 0.9790 |
| ROC-AUC | 0.9998 |

### Confusion Matrix

```
Confusion Matrix
         0     1     2     3     4     5     6     7     8     9
 0 |   968     0     2     0     0     3     2     1     2     2
 1 |     0  1130     2     0     0     1     1     0     1     0
 2 |     3     6  1006     1     2     0     1     7     6     0
 3 |     1     0     1   979     0    19     0     5     1     4
 4 |     0     0     1     0   967     0     1     1     2    10
 5 |     1     0     0     0     0   885     1     0     2     3
 6 |     5     3     0     0     4    10   935     0     1     0
 7 |     0     3     9     2     0     1     0  1002     2     9
 8 |     6     1     3     6     1     6     4     3   937     7
 9 |     2     5     1     0     7     6     0     5     2   981
```

---

## Model 4 (Best Model)

### Evaluation Metrics

| Metric | Value |
|--------|------:|
| Loss | 0.0565 |
| Accuracy | **0.9817** |
| Precision | **0.9818** |
| Recall | **0.9817** |
| F1 Score | **0.9817** |
| ROC-AUC | **0.9998** |

### Confusion Matrix

```
Confusion Matrix
         0     1     2     3     4     5     6     7     8     9
 0 |   974     0     1     0     0     2     0     1     2     0
 1 |     0  1133     0     0     0     0     1     0     1     0
 2 |     5     4   998     6     1     0     1     7    10     0
 3 |     1     1     0   991     0     8     0     4     2     3
 4 |     1     0     0     0   968     0     2     3     2     6
 5 |     2     0     0     1     0   884     1     2     2     0
 6 |     8     1     0     1     1     5   938     1     3     0
 7 |     2     1     7     4     0     0     0  1006     2     6
 8 |     6     0     2     3     2     6     0     2   950     3
 9 |     5     5     0     4     6     5     0     6     3   975
```