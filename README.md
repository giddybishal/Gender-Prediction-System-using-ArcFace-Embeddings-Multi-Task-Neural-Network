# Gender Classification using ArcFace Embeddings + Multi-Task MLP

A deep learning pipeline for **gender classification** (primary task) and **age-group prediction** (secondary task) using:

- ArcFace facial embeddings
- InsightFace face detection/alignment
- Multi-task MLP architecture
- PyTorch
- ONNXRuntime GPU
- Optuna hyperparameter tuning

The project combines the **UTKFace** dataset and the **Face-Age-Gender Dataset** from Kaggle into a unified training pipeline.

---

# 🚀 Features

- GPU-accelerated embedding extraction using InsightFace
- ArcFace 512-dimensional face embeddings
- Multi-task learning:
  - Gender Classification
  - Age Group Classification
- Optuna-based hyperparameter optimization
- Confidence score output during inference
- No-face detection handling
- CPU/GPU inference support
- Batch embedding generation and caching

---

# 🧠 Model Architecture

```text
Input Image
    ↓
Face Detection + Alignment (InsightFace)
    ↓
ArcFace Embedding (512D)
    ↓
Shared MLP Backbone
    ├── Gender Head
    └── Age Head
```

---

# 📦 Datasets Used

## 1. UTKFace

Filename format:

```text
age_gender_race_date.jpg
```

Used fields:
- age
- gender

---

## 2. Face-Age-Gender Dataset (Kaggle)

CSV-based dataset containing:
- image path
- age
- gender

---

# 🏷️ Age Groups

```python
[
    "<18",
    "18-24",
    "25-34",
    "35-44",
    "45-54",
    "55-64",
    "65+"
]
```

---

# ⚙️ Technologies Used

- Python
- PyTorch
- InsightFace
- ONNXRuntime-GPU
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Optuna

---

# 🔥 GPU Setup

ONNXRuntime GPU was used to ensure CUDA acceleration during:
- face detection
- embedding extraction

Verify GPU providers:

```python
import onnxruntime as ort

print(ort.get_available_providers())
```

Expected output:

```python
['TensorrtExecutionProvider',
 'CUDAExecutionProvider',
 'CPUExecutionProvider']
```

---

# 🧪 Training Pipeline

## Step 1 — Combine Datasets

Both datasets are:
- cleaned
- standardized
- merged into one dataframe

---

## Step 2 — Stratified Split

Dataset split:
- Train
- Validation
- Test

Stratified using:

```python
age_label
```

to preserve balanced age-group distribution.

---

## Step 3 — Face Embedding Extraction

Using:

```python
FaceAnalysis(name="buffalo_l")
```

Each image:
1. Detects face
2. Aligns face
3. Extracts ArcFace embedding

Output:
```text
512-dimensional embedding vector
```

---

## Step 4 — Save Embeddings

Embeddings cached using:

```python
np.savez(...)
```

This avoids recomputing expensive ArcFace embeddings every training run.

---

## Step 5 — Multi-Task MLP Training

Shared feature extractor:
- Linear layers
- BatchNorm
- GELU activation
- Dropout

Two output heads:
- Gender classification
- Age classification

---

## Step 6 — Hyperparameter Optimization

Optuna used to tune:
- hidden dimension
- dropout
- learning rate
- weight decay

---

# 📊 Final Results

## Validation Performance

| Metric | Score |
|---|---|
| Gender Accuracy | 95.92% |
| Age Accuracy | 49.90% |

---

## Test Performance

| Metric | Score |
|---|---|
| Gender Accuracy | 95.67% |
| Age Accuracy | 48.17% |

---

# 📈 Gender Classification Report

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Male | 0.90 | 0.92 | 0.91 |
| Female | 0.98 | 0.97 | 0.97 |

Overall Accuracy:

```text
96%
```

---

# 🖼️ Sample Inference Results

> Add prediction collage / visualization here.

---

# 🖼️ Inference Features

The inference pipeline supports:

- Single image inference
- Folder inference
- Confidence scores
- No-face detection
- Inference timing

Example output:

```text
Prediction: Female
Confidence: 0.9912
Inference Time: 34.5 ms
```

---

# 🚨 No Face Detection Handling

Inference returns status enums:

```python
SUCCESS
NO_FACE_DETECTED
IMAGE_LOAD_FAILED
```

This improves production robustness.

---

# ⏱️ Performance Notes

Most inference latency comes from:
- face detection
- ArcFace embedding extraction

The MLP classifier itself is extremely lightweight.

---

# 📂 Project Structure

```text
project/
│
├── train_embeddings.npz
├── val_embeddings.npz
├── test_embeddings.npz
│
├── gender_age_mlp.pth
│
├── train.ipynb
├── inference.ipynb
│
└── README.md
```

---

# 💾 Model Saving

```python
torch.save(model.state_dict(), "gender_age_mlp.pth")
```

---

# 🚀 Future Improvements

Potential future improvements include:

- Lightweight embedding models (`buffalo_s`)
- End-to-end gender classification models
- Better age classification balancing
- Confidence calibration
- ONNX/TensorRT deployment
- Real-time webcam inference

---

# 👨‍💻 Authors

Developed as part of a deep learning and computer vision internship/research project focused primarily on robust gender classification using ArcFace embeddings and multi-task learning.