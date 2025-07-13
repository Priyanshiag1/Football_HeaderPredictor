# ⚽ Football Header Predictor (Regularization in Deep Learning)

This project is a demonstration of how **regularization techniques** like **L2 Regularization** and **Dropout** can improve the generalization of neural networks and reduce overfitting. The model predicts whether a **French football player** will win the ball header based on its 2D landing coordinates.

---

## 📌 Problem Statement

> You have just been hired as an AI expert by the French Football Corporation.  
> Based on previous match data, you must predict **whether the French team will win the header**, given where the ball lands on the field.

---

## 🧠 Techniques Applied

- ✅ Deep Neural Network (3-layer)
- ✅ Sigmoid + ReLU activations
- ✅ Binary classification
- ✅ Applied **Regularization**:
  - **L2 Regularization** (penalizing large weights)
  - **Dropout** (randomly deactivating neurons during training)

---

## 📊 Results Comparison

| Model Variant           | Training Accuracy | Test Accuracy |
|-------------------------|-------------------|---------------|
| Without Regularization  | 94.79%            | 91.5%         |
| With L2 Regularization  | 93.83%            | 93.0%         |
| With Dropout            | 92.89%            | **95.0%**     |

✅ **Dropout** provided the best generalization on the test set.

---

## 🖼️ Streamlit Web App

A minimal interactive web app was built using **Streamlit** to visualize predictions.

### Features:
- Users can input coordinates where the ball lands
- App predicts if a French player is likely to win the header
- Visual feedback on the football field image

