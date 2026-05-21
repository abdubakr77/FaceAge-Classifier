# FaceAge-Classifier

Age group classification from face images using Transfer Learning (ResNet50 & DenseNet121).

---

## Overview

This project trains two pretrained CNN models to classify face images into one of three age groups:

| Class | Age Range |
|-------|-----------|
| 14-24 | Young adults |
| 25-40 | Adults |
| 41-70 | Older adults |

The dataset contains ~32,000 labeled face images. Both models were trained using full fine-tuning with a low learning rate (1e-5) and early stopping.

---

## Results

| Model | Val Accuracy |
|-------|-------------|
| ResNet50 | ~68% |
| DenseNet121 | ~70% |

The main source of confusion is between adjacent age groups (e.g. 25-40 vs 41-70), which is expected since age boundaries are not visually sharp.

---

## Datasets

[Age Detection - Face Recognition Dataset (Kaggle)](https://www.kaggle.com/datasets/trainingdatapro/age-detection-human-faces-18-60-years)
[Face-Age-Gender Dataset](https://www.kaggle.com/datasets/aadyasingh55/face-age-gender-dataset/data)

The original dataset has 5 age groups (18-20, 21-30, 31-40, 41-50, 51-60). In this project the classes were re-mapped into 3 broader groups using a secondary dataset with raw age labels.

---

## Project Structure

```
FaceAge-Classifier/
    Face_Age_Classifier.ipynb   main notebook
    README.md
    Results/
        ResNet_LOSS_Curves.png
        ResNet_confusion_matrix.png
        DenseNet_ACCURACY_LOSS_Curves.png
        DenseNet_confusion_matrix.png
        test_set_predictions_ResNet.png
        test_set_predictions_DenseNet.png
```

---

## How to Run

**On Kaggle:**

1. Add the dataset to your notebook.
2. Update `MAIN_PATH` to `/kaggle/input/age-detection-human-faces-18-60-years`.
3. Run all cells in order.

**Locally:**

```bash
pip install torch torchvision scikit-learn pandas matplotlib seaborn tqdm pillow opencv-python
```

Then open `Face_Age_Classifier.ipynb` and update `MAIN_PATH` to your local dataset path.

---

## Approach

- **Pretrained models**: ResNet50 and DenseNet121 with ImageNet weights
- **Fine-tuning**: All layers trained (full fine-tuning) at a very low LR (1e-5)
- **Classifier head**: Original head replaced with Dropout(0.3) + Linear(NUM_CLASSES)
- **Augmentation**: Random horizontal flip, rotation (±20°), color jitter
- **Optimizer**: AdamW
- **Loss**: Cross-entropy
- **Early stopping**: Patience = 5 epochs based on val loss

---

## Key Observations

- DenseNet121 slightly outperforms ResNet50 on this task.
- The 25-40 class is the hardest to predict accurately (confused with both adjacent groups).
- The 14-24 and 41-70 classes are easier to separate.
- With ~32K images and transfer learning, both models converge within 10 epochs.