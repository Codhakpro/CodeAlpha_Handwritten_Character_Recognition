import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Codhak Pro — Handwritten Digit AI",
    page_icon="✍️",
    layout="centered"
)


# ============================================================
# MODEL
# ============================================================

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),

            nn.Dropout(0.25),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = CNN().to(device)

model.load_state_dict(
    torch.load(
        "handwritten_character_cnn.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    # Convert to grayscale
    image = image.convert("L")

    # Invert colors
    image = ImageOps.invert(image)

    # Resize to MNIST size
    image = image.resize((28, 28))

    # Convert to tensor
    transform = transforms.ToTensor()

    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor.to(device)


# ============================================================
# PREDICTION
# ============================================================

def predict_digit(image):

    image_tensor = preprocess_image(image)

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

        prediction = torch.argmax(
            probabilities
        ).item()

        confidence = probabilities[prediction].item()

    return prediction, confidence, probabilities


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 17px;
        margin-bottom: 35px;
    }

    .prediction {
        text-align: center;
        font-size: 110px;
        font-weight: 800;
        line-height: 1;
        margin: 10px;
    }

    .confidence {
        text-align: center;
        font-size: 20px;
        color: #9ca3af;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">✍️ Handwritten Digit AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Upload a handwritten digit and let the CNN identify it.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"],
    help="Upload an image containing a handwritten digit from 0 to 9."
)


# ============================================================
# IMAGE + PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded handwriting",
        width=300
    )

    st.divider()

    if st.button(
        "🧠 Predict Digit",
        type="primary",
        use_container_width=True
    ):

        prediction, confidence, probabilities = predict_digit(
            image
        )

        st.markdown(
            "### Prediction"
        )

        st.markdown(
            f'<div class="prediction">{prediction}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="confidence">'
            f'Confidence: {confidence * 100:.2f}%'
            f'</div>',
            unsafe_allow_html=True
        )

        st.divider()

        st.markdown("### 📊 Prediction probabilities")

        for digit, probability in enumerate(probabilities):

            st.write(
                f"**{digit}** — {probability.item() * 100:.2f}%"
            )

            st.progress(
                float(probability.item())
            )


# ============================================================
# INFORMATION
# ============================================================

st.divider()

st.markdown(
    """
    ### About the model

    This application uses a **Convolutional Neural Network (CNN)**
    trained on the **MNIST handwritten digit dataset**.

    The model recognizes digits from **0 to 9**.

    > This is an educational machine-learning demonstration.
    """
)