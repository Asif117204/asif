<div align="center">

# ✊ ✋ ✌️
## Rock–Paper–Scissors Image Classification using CNN (PyTorch)

</div>

## 1️⃣ Introduction

This project presents an end-to-end implementation of a **Convolutional Neural Network (CNN)** for **Rock–Paper–Scissors (RPS)** hand gesture classification using **PyTorch**.

The model is trained using two image-folder datasets and evaluated on an unseen validation set, real-world smartphone images, and sample images from the training dataset. Visual error analysis is also performed to understand model behavior and limitations.

The goal of this project is not only to achieve high accuracy but also to examine how the model performs on images outside the training distribution.

---

## 2️⃣ Datasets Used

### 2.1 Training Datasets

The project loads two image-folder datasets:

- `train_dataset1`
- `train_dataset2`

The two datasets are combined using `ConcatDataset`.

- **Classes:** Paper, Rock, Scissors
- **Total images:** 4,708
- **Training split:** 3,766 images (80%)
- **Validation split:** 942 images (20%)

The classes detected by the notebook are:

```text
['paper', 'rock', 'scissors']
```

---

### 2.2 Custom Smartphone Dataset

- **Source:** Author's smartphone camera
- **Purpose:** Real-world evaluation
- **Transform:** Resize, tensor conversion, and ImageNet normalization
- **Augmentation:** Not applied during testing

This dataset is used to examine how well the trained CNN generalizes to real-world images.

---

## 3️⃣ Data Preprocessing

### 3.1 Training Transform (With Data Augmentation)

The training pipeline uses:

- Resize to **224 × 224**
- Random horizontal flip
- Random rotation up to 30°
- Color jitter for brightness, contrast, saturation, and hue
- Convert to tensor
- Normalize using ImageNet statistics

```text
Mean = [0.485, 0.456, 0.406]
Std  = [0.229, 0.224, 0.225]
```

### 3.2 Validation / Test / Phone Transform

- Resize to **224 × 224**
- Convert to tensor
- Apply ImageNet normalization
- No data augmentation

---

## 4️⃣ CNN Architecture

The CNN consists of **three convolutional blocks** followed by fully connected classifier layers.

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
```

### 4.2 Architecture Summary

| Component | Description |
| ---------------------- | -------------------- |
| Convolution Layers | Feature extraction |
| ReLU | Non-linearity |
| MaxPooling | Spatial downsampling |
| Dropout (0.5) | Overfitting control |
| Fully Connected Layers | Final classification |

---

## 5️⃣ Training Configuration

- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Batch Size:** 64
- **Epochs:** 10
- **Device:** CUDA if available, otherwise CPU

The model is explicitly moved to the selected device:

```python
model = CNN(num_classes=3).to(device)
```

---

## 6️⃣ Experimental Results

### 6.1 Training & Validation Curves

<img width="1324" height="575" alt="Image" src="https://github.com/user-attachments/assets/342a1faa-25d1-423e-99be-96b3975829ef" />

The notebook records the following validation/training accuracy progression:

| Epoch | Training Accuracy | Validation Accuracy |
|------:|------------------:|--------------------:|
| 1 | 69.41% | 94.16% |
| 2 | 94.29% | 96.82% |
| 3 | 96.95% | 97.56% |
| 4 | 98.17% | 96.92% |
| 5 | 98.65% | 98.51% |
| 6 | 99.42% | 98.62% |
| 7 | 99.68% | **98.73%** |
| 8 | 99.55% | 98.20% |
| 9 | **99.79%** | 98.51% |
| 10 | 99.42% | 98.30% |

The best recorded validation accuracy is approximately **98.73%**, while the highest training accuracy is approximately **99.79%**.

---

### 6.2 Confusion Matrix (Validation Set)

![Confusion Matrix](assets/confusion_matrix.png)

The confusion matrix shows the classification behavior of the CNN across the three classes:

- Paper
- Rock
- Scissors

---

### 6.3 Predictions on Standard Dataset Samples

<img width="1297" height="710" alt="Image" src="https://github.com/user-attachments/assets/b7484fb9-7fc8-4d44-8aa6-03f3bc35e600" />

The visualization contains **10 randomly selected samples from the training dataset**.

For each image:

- `T:` = True class
- `P:` = Predicted class
- Percentage = Model confidence

The notebook uses the model's softmax probabilities to calculate the prediction confidence.

> **Note:** These are samples from the combined training dataset. The notebook does not separately label these 10 samples as Kaggle versus TensorFlow datasets.

---

### 6.4 Predictions on Real Smartphone Images

![Phone Image Predictions](assets/phone_predictions.png)

The trained model is also tested on the custom smartphone images.

The prediction visualization displays:

- Input smartphone image
- Predicted class
- Prediction confidence

This evaluation helps assess real-world generalization beyond the training/validation distribution.

---

## 7️⃣ Real-World Generalization

The project compares model behavior on standard dataset samples and custom smartphone images.

A high validation accuracy demonstrates strong performance on the validation distribution, but real-world smartphone images can still expose differences in:

- Background
- Lighting
- Hand scale
- Camera quality
- Image composition
- Orientation

These differences can create a **domain shift** between training data and real-world inputs.

---

## 8️⃣ Key Insight

> High validation accuracy does **not automatically guarantee real-world robustness**.

This project demonstrates why evaluating a CNN on both standard validation data and custom real-world images is important.

---

## 9️⃣ Future Improvements

- Apply **Transfer Learning** using models such as ResNet or MobileNet
- Collect more real-world smartphone images
- Increase background and lighting diversity
- Use stronger data augmentation
- Fine-tune the model using representative real-world data
- Analyze precision, recall, and F1-score in addition to accuracy

---

## 🔟 Conclusion

This project demonstrates a complete CNN-based image classification pipeline using PyTorch:

- Dataset loading and train/validation splitting
- Data augmentation and normalization
- CNN model development
- GPU/CPU device support
- Model training and validation
- Training/validation curve visualization
- Confusion matrix analysis
- Standard dataset prediction visualization
- Real-world smartphone image evaluation

The results show strong validation performance while also emphasizing the importance of testing models on data that may differ from the training distribution.

---

## 👨‍🎓 Author

**Md. Asif Mia**  
ID: 220128  
Department of CSE, JUST

---

## 📎 Acknowledgements

- Rock–Paper–Scissors image datasets used in the project
- PyTorch
- Torchvision
- Matplotlib
- Seaborn

---

### ✅ Final Note

This work highlights the importance of **critical model evaluation**, not just accuracy metrics, when developing deep learning systems for real-world applications.
