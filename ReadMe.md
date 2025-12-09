# GoogLeNet for Placental Histopathology Image Classification

This project implements and evaluates a customized GoogLeNet architecture for multi-class classification of placental tissue microscopy images. The workflow includes dataset preparation, preprocessing, transfer learning, grid search over hyperparameters, extended fine-tuning, class-weighted training, and systematic comparison of model variants.

---

## 1. Problem and Data

The goal is to classify four types of placental tissue from microscopic images:

- Chorionic villi  
- Decidual tissue  
- Hemorrhage  
- Trophoblastic tissue  

The dataset is organized as:

```text
data/
└── POC_Dataset/
    ├── Training/
    │   ├── Chorionic_villi/
    │   ├── Decidual_tissue/
    │   ├── Hemorrhage/
    │   └── Trophoblastic_tissue/
    └── Testing/
        ├── Chorionic_villi/
        ├── Decidual_tissue/
        ├── Hemorrhage/
        └── Trophoblastic_tissue/
````

Each subfolder contains the corresponding class images.
Training data is internally split into train/validation using `random_split`.

---

## 2. Model: GoogLeNet Adaptation

The project uses PyTorch’s GoogLeNet implementation with ImageNet-pretrained weights:

```python
def create_googlenet(num_classes=4, pretrained=True):
    weights = models.GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.googlenet(weights=weights, aux_logits=True)

    # main classifier
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # auxiliary classifiers
    if model.aux1 is not None:
        model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, num_classes)
    if model.aux2 is not None:
        model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_classes)

    return model
```

Differences vs original GoogLeNet:

* 1000-class heads are replaced with 4-class heads (main + both auxiliaries).
* ImageNet weights are reused for convolutional/Inception layers (transfer learning).
* Core architecture (Inception blocks) remains unchanged; only classification layers are task-specific.

---

## 3. Training Strategy

### 3.1 Baseline

* Optimizer: SGD (lr = 0.0001, momentum = 0.9, weight decay = 0.0)
* Aux loss weight: 0.4 (main + aux1 + aux2)
* Epochs: 20
* Loss: standard cross-entropy

### 3.2 Longer Training

The model was then trained for **40 epochs** with the same hyperparameters to allow full convergence.

### 3.3 Class-Weighted Loss

To address class imbalance, the loss was changed to a **class-weighted CrossEntropyLoss**:

```python
class_counts = np.array([1391, 926, 1138, 700], dtype=np.float32)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
```

This improved recall and F1, especially for Decidual and Trophoblastic tissue.

### 3.4 Data Augmentation

Moderate augmentation is used during training:

```python
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])
```

More aggressive augmentations (RandomResizedCrop, heavy blur/sharpness) were tested, but slightly degraded performance, so the final model uses the more conservative pipeline above.

---

## 4. Results and Final Model

### 4.1 Comparison of Model Variants

```markdown
| Model Variant                              | Test Loss | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|--------------------------------------------|-----------|----------|-----------------|--------------|----------|
| 20 epochs                                  | 0.8592    | 0.7445   | 0.8067          | 0.7276       | 0.7241   |
| 40 epochs                                  | 0.5466    | 0.8365   | 0.8513          | 0.8270       | 0.8276   |
| 40 epochs + class weights (final model)    | 0.5457    | 0.8524   | 0.8668          | 0.8438       | 0.8454   |
| 40 epochs + class weights + strong aug     | 0.5762    | 0.8332   | 0.8485          | 0.8245       | 0.8221   |
```

### 4.2 Final Model Metrics (40 epochs + class weights)

* Accuracy: **0.8524**
* Macro F1: **0.8454**

Per-class performance:

| Class                | Precision | Recall | F1     |
| -------------------- | --------- | ------ | ------ |
| Chorionic villi      | 0.8847    | 0.9641 | 0.9227 |
| Decidual tissue      | 0.8889    | 0.6189 | 0.7297 |
| Hemorrhage           | 0.7633    | 0.9572 | 0.8493 |
| Trophoblastic tissue | 0.9302    | 0.8348 | 0.8799 |

---

## 5. Repository Structure

```text
.
├── data/
│   └── POC_Dataset/
│       ├── Training/
│       │   ├── Chorionic_villi/
│       │   ├── Decidual_tissue/
│       │   ├── Hemorrhage/
│       │   └── Trophoblastic_tissue/
│       └── Testing/
│           ├── Chorionic_villi/
│           ├── Decidual_tissue/
│           ├── Hemorrhage/
│           └── Trophoblastic_tissue/
│
├── final_model/
│   ├── best_40_googlenetv3.pth          # selected final checkpoint
│   └── conf_matrix/
│       ├── cm_googlenetv3_raw.png       # confusion matrix (counts)
│       └── cm_googlenetv3_norm.png      # confusion matrix (normalized)
│
├── test_data/
│   ├── best_googlenet_lr=...pth         # all tested model checkpoints from grid search
│   ├── cm_*.png                         # confusion matrices for each tested model
│   ├── googlenet_experiments_results.csv# validation metrics for all experiments
│   ├── model_result.csv / model_results.csv
│   └── model_comparison.png             # bar plot comparing test accuracy of all models
│
└── GoogLeNet.ipynb                      # main training, evaluation and analysis notebook
```

* `test_data/` contains the artifacts of the hyperparameter search:
  all best checkpoints, their confusion matrices, CSV logs with validation/test metrics, and a comparison plot.
* `final_model/` stores the **single selected final model** and its confusion matrices for easy reuse and reporting.

---

## 6. Conclusion

The project shows that a carefully adapted GoogLeNet with ImageNet pretraining, longer fine-tuning, class-weighted loss, and moderate augmentation can achieve strong, balanced performance on a multi-class placental histopathology dataset. The final model significantly improves over the initial baseline, particularly in recall and F1 for clinically more challenging tissue classes.
