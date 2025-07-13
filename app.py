import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load dropout model directly
@st.cache_resource
def load_dropout_model():
    with open("model_dropout.pkl", "rb") as f:
        return pickle.load(f)

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def relu(Z):
    return np.maximum(0, Z)

# Forward propagation
def forward(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    return A3

# Main app
st.set_page_config(page_title="Football Header Predictor", layout="centered")
st.title("⚽ Football Header Predictor (Dropout Model)")
st.markdown("Enter a point on the field to check if a **French Player** is likely to win the header.")

# Load model
parameters = load_dropout_model()

# Input coordinates
x1 = st.slider("x1 coordinate (-1 to 1)", -1.0, 1.0, 0.0, step=0.01)
x2 = st.slider("x2 coordinate (-1 to 1)", -1.0, 1.0, 0.0, step=0.01)

if st.button("Predict"):
    X_input = np.array([[x1], [x2]])
    prob = forward(X_input, parameters)
    label = int(prob > 0.5)

    st.markdown(f"### Prediction: {'🔵 French Player' if label == 1 else '🔴 Other Player'}")
    st.markdown(f"#### Confidence: `{float(prob) * 100:.2f}%`")

    # Field image
    img = mpimg.imread("field_kiank.png")
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.scatter((x1 + 1) * img.shape[1] / 2, (1 - x2) * img.shape[0] / 2, c='yellow', s=100, label="Ball Landing")
    ax.legend()
    ax.axis('off')
    st.pyplot(fig)

st.markdown("---")
st.caption("Model: Trained using Dropout Regularization")
