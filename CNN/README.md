
<div align="center">

# ✊ ✋ ✌️  
## Rock–Paper–Scissors Image Classification using CNN (PyTorch)

</div>


## 1️⃣ Introduction

This project presents an end-to-end implementation of a **Convolutional Neural Network (CNN)** for **Rock–Paper–Scissors (RPS)** hand gesture classification using **PyTorch**.  
The model is trained on **two standard benchmark datasets** and evaluated on both **in-distribution data** and **real-world smartphone images** to analyze generalization performance.

The goal of this project is not only to achieve high accuracy but also to **study the limitations of CNN models when deployed in real-world environments**.

---

## 2️⃣ Datasets Used

### 2.1 Kaggle Rock–Paper–Scissors Dataset

- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors  
- **Classes:** Rock, Paper, Scissors  
- **Image Type:** RGB  
- **Characteristics:**
  - Clean background
  - Centered hand gestures
  - Consistent lighting

This dataset provides a controlled environment suitable for supervised CNN training.

---

### 2.2 TensorFlow Rock–Paper–Scissors Dataset

- **Source:** TensorFlow Datasets (TFDS)  
- **Characteristics:**
  - Larger dataset size
  - Slight pose variations
  - Still captured under controlled conditions

Both Kaggle and TensorFlow datasets belong to the **same visual domain**, allowing the CNN to learn highly discriminative features.

---

### 2.3 Custom Smartphone Dataset (Evaluation Only)

- **Source:** Author’s smartphone camera  
- **Number of Images:** 10  
- **Conditions:**
  - proper lighting
  - White clean backgrounds
  - Different camera angles
  - Varying hand scales

This dataset is used **only for testing** to evaluate real-world generalization.

---

## 3️⃣ Data Preprocessing

### 3.1 Training Transform (With Data Augmentation)

- Resize to **224 × 224**
- Random horizontal flip
- Random rotation
- Color jitter (brightness & contrast)
- Convert to tensor
- Normalize using ImageNet statistics

```

Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]

````

### 3.2 Validation & Phone Image Transform

- Resize to **224 × 224**
- Convert to tensor
- Apply the same normalization
- No data augmentation

---

## 4️⃣ CNN Architecture

The CNN consists of **three convolutional blocks** followed by a **fully connected classifier**.

### 4.1 Model Implementation

```python
class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
````

### 4.2 Architecture Summary

| Component              | Description          |
| ---------------------- | -------------------- |
| Convolution Layers     | Feature extraction   |
| ReLU                   | Non-linearity        |
| MaxPooling             | Spatial downsampling |
| Dropout (0.5)          | Overfitting control  |
| Fully Connected Layers | Final classification |

---

## 5️⃣ Training Configuration

* **Loss Function:** CrossEntropyLoss
* **Optimizer:** Adam
* **Batch Size:** 64
* **Epochs:** 10
* **Device:** GPU (CUDA) if available

---

## 6️⃣ Experimental Results

### 6.1 Training & Validation Curves

<img width="1181" height="547" alt="Screenshot From 2025-12-25 23-30-26" src="https://github.com/user-attachments/assets/d7d792f0-707b-439f-93e9-2701a3aa0c90" />


* Training accuracy reaches **~100%**
* Validation accuracy stabilizes around **98–99%**
* Indicates strong in-distribution learning

---

### 6.2 Confusion Matrix (Validation Set)

<img width="645" height="547" alt="Screenshot From 2025-12-25 23-30-41" src="https://github.com/user-attachments/assets/1d2fe38f-cb02-4116-8ceb-487384d1fff5" />


* Minimal class confusion
* Most errors occur between **scissors → paper**

---

### 6.3 Predictions on Standard Datasets

<img width="804" height="405" alt="Screenshot From 2025-12-25 23-31-42" src="https://github.com/user-attachments/assets/c8b9ebce-c177-4fb4-8eb1-d80414299b74" />

* Near-perfect confidence
* Correct predictions for Kaggle and TensorFlow datasets

---

### 6.4 Predictions on Real Smartphone Images
<img width="1169" height="750" alt="Screenshot From 2025-12-25 23-31-18" src="https://github.com/user-attachments/assets/b9bbfe99-751d-4fea-9b70-f9f4e1adba2b" />

* Several incorrect predictions
* Reduced confidence scores
* Clear performance degradation

---

## 7️⃣ Why Does the Model Fail on Real Phone Images?

Despite excellent performance on standard datasets, the model struggles on real phone images due to the following reasons:

### 7.1 Domain Shift (Primary Reason)

Training images are captured in controlled environments, while phone images introduce:

* Different backgrounds
* lighting variations
* Camera noise and blur
* Scale and orientation differences

This creates a **distribution mismatch** between training and testing data.

---

### 7.2 Background Bias

The CNN unintentionally learns background-related features instead of purely hand geometry.
When the background changes, predictions become unreliable.

---

### 7.3 Limited Model Capacity

The CNN is trained from scratch and lacks the representational power of large pretrained models.

---

### 7.4 Small Real-World Sample Size

Only **10 phone images** are insufficient for real-world adaptation.

---

## 8️⃣ Key Insight

> High validation accuracy does **not guarantee real-world robustness**.

This project demonstrates a common deep learning limitation:
**CNNs generalize poorly outside their training distribution without domain adaptation or transfer learning.**

---

## 9️⃣ Future Improvements

* Apply **Transfer Learning** (ResNet, MobileNet)
* Collect more real-world images
* Use background randomization
* Perform fine-tuning on phone images

---

## 🔟 Conclusion

This project demonstrates:

* A complete CNN-based image classification pipeline
* Strong performance on two benchmark datasets
* Clear analysis of real-world generalization failure

The results emphasize the importance of **data diversity and domain alignment** in deep learning systems.

---

## 👨‍🎓 Author

**Adnan Zaman Niloy** ,
ID: 210142, Dept: CSE(JUST)

---

## 📎 Acknowledgements

* Kaggle Rock–Paper–Scissors Dataset
* TensorFlow Datasets
* PyTorch & Torchvision

---

### ✅ Final Note

This work highlights the importance of **critical model evaluation**, not just accuracy metrics, which is essential for real-world deep learning applications.
