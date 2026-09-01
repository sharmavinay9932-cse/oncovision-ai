# ============================================================
# ONCOVISION AI
# Multi-Cancer Detection System
# Breast Cancer -> RBF SVM
# Skin Cancer   -> Fine-Tuned ResNet18 CNN
# ============================================================

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18

from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="OncoVision AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

SKIN_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "skin",
    "best_resnet18_finetuned.pth"
)

SVM_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "svm_model.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

CONFIG_PATH = os.path.join(
    MODEL_DIR,
    "config.pkl"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background-color: #f5f7fb;
    }

    /* Header */
    .main-header {
        background: white;
        padding: 25px 35px;
        border-radius: 18px;
        margin-bottom: 25px;
        border: 1px solid #e5e7eb;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 18px;
    }

    .brand-icon {
        width: 60px;
        height: 60px;
        border-radius: 16px;
        background: #172554;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
    }

    .brand-title {
        font-size: 32px;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
    }

    .brand-subtitle {
        color: #64748b;
        margin-top: 4px;
    }

    /* Cards */
    .card {
        background: white;
        padding: 28px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        margin-bottom: 25px;
    }

    /* Section title */
    .section-title {
        color: #0f172a;
        font-size: 28px;
        font-weight: 750;
        margin-bottom: 5px;
    }

    .section-description {
        color: #64748b;
        font-size: 16px;
    }

    /* Result */
    .result-card {
        background: white;
        padding: 30px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        margin-top: 25px;
    }

    .result-title {
        font-size: 30px;
        font-weight: 800;
        color: #0f172a;
    }

    .probability {
        font-size: 28px;
        font-weight: 800;
    }

    /* Disclaimer */
    .disclaimer {
        background: #fff8e7;
        border: 1px solid #f2cf75;
        padding: 20px;
        border-radius: 15px;
        margin-top: 30px;
        color: #713f12;
    }

    .disclaimer-title {
        font-weight: 800;
        font-size: 18px;
        margin-bottom: 5px;
    }

    /* Info boxes */
    .info-box {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        padding: 15px;
        border-radius: 12px;
        color: #1e3a8a;
        margin-bottom: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">

        <div class="brand">

            <div class="brand-icon">
                🧬
            </div>

            <div>

                <div class="brand-title">
                    OncoVision AI
                </div>

                <div class="brand-subtitle">
                    AI-powered multi-cancer detection research system
                </div>

            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD BREAST CANCER MODEL
# ============================================================

@st.cache_resource
def load_breast_model():

    with open(SVM_MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    with open(CONFIG_PATH, "rb") as file:
        config = pickle.load(file)

    features = config["features"]

    threshold = config.get(
        "threshold",
        0.62
    )

    return model, scaler, features, threshold


# ============================================================
# LOAD SKIN CNN
# ============================================================

@st.cache_resource
def load_skin_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # --------------------------------------------------------
    # ResNet18
    # --------------------------------------------------------

    model = resnet18(
        weights=None
    )

    # Your model is binary classification
    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    # --------------------------------------------------------
    # Load trained weights
    # --------------------------------------------------------

    checkpoint = torch.load(
        SKIN_MODEL_PATH,
        map_location=device
    )

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint


    # Remove DataParallel prefix if present
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[7:]

        cleaned_state_dict[new_key] = value


    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.to(device)

    model.eval()

    return model, device


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

skin_transform = transforms.Compose(
    [

        transforms.Resize(
            (224, 224)
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],

            std=[
                0.229,
                0.224,
                0.225
            ]
        )

    ]
)


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    image_tensor,
    device
):

    activations = []
    gradients = []


    # --------------------------------------------------------
    # Target layer
    # --------------------------------------------------------

    target_layer = model.layer4[-1]


    # --------------------------------------------------------
    # Forward hook
    # --------------------------------------------------------

    def forward_hook(
        module,
        input,
        output
    ):

        activations.append(
            output.detach()
        )


    # --------------------------------------------------------
    # Backward hook
    # --------------------------------------------------------

    def backward_hook(
        module,
        grad_input,
        grad_output
    ):

        gradients.append(
            grad_output[0].detach()
        )


    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )


    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    model.zero_grad()


    output = model(
        image_tensor.to(device)
    )


    prediction = output.argmax(
        dim=1
    ).item()


    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    score = output[
        0,
        prediction
    ]


    score.backward()


    # --------------------------------------------------------
    # Remove hooks
    # --------------------------------------------------------

    forward_handle.remove()
    backward_handle.remove()


    # --------------------------------------------------------
    # Get activation + gradient
    # --------------------------------------------------------

    activation = activations[0][0]

    gradient = gradients[0][0]


    # --------------------------------------------------------
    # Global average pooling
    # --------------------------------------------------------

    weights = gradient.mean(
        dim=(1, 2)
    )


    # --------------------------------------------------------
    # Weighted feature maps
    # --------------------------------------------------------

    cam = torch.zeros(
        activation.shape[1:],
        device=device
    )


    for i, weight in enumerate(weights):

        cam += (
            weight *
            activation[i]
        )


    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    cam = torch.relu(
        cam
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    cam -= cam.min()

    if cam.max() > 0:

        cam /= cam.max()


    cam = cam.cpu().numpy()

    return cam, prediction


# ============================================================
# OVERLAY GRAD-CAM
# ============================================================

def create_gradcam_overlay(
    original_image,
    cam
):

    # Resize CAM
    cam_image = Image.fromarray(
        np.uint8(cam * 255)
    )

    cam_image = cam_image.resize(
        original_image.size
    )


    cam_array = np.array(
        cam_image
    ) / 255.0


    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )


    ax.imshow(
        original_image
    )

    ax.imshow(
        cam_array,
        cmap="jet",
        alpha=0.45
    )

    ax.axis("off")

    fig.tight_layout(
        pad=0
    )


    return fig


# ============================================================
# LOAD MODELS
# ============================================================

try:

    breast_model, breast_scaler, breast_features, breast_threshold = (
        load_breast_model()
    )

    breast_loaded = True

except Exception as error:

    breast_loaded = False

    breast_error = str(error)


try:

    skin_model, skin_device = (
        load_skin_model()
    )

    skin_loaded = True

except Exception as error:

    skin_loaded = False

    skin_error = str(error)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🧬 OncoVision AI"
    )

    st.markdown(
        "### Cancer Detection"
    )


    cancer_type = st.radio(
        "Select Cancer Type",
        [
            "🩺 Breast Cancer",
            "🔬 Skin Cancer"
        ]
    )


    st.divider()


    st.markdown(
        "### Model Status"
    )


    if breast_loaded:

        st.success(
            "Breast SVM: Loaded"
        )

    else:

        st.error(
            "Breast SVM: Error"
        )


    if skin_loaded:

        st.success(
            "Skin ResNet18: Loaded"
        )

    else:

        st.error(
            "Skin ResNet18: Error"
        )


    st.divider()


    st.caption(
        "Research & educational project"
    )


# ============================================================
# BREAST CANCER PAGE
# ============================================================

if cancer_type == "🩺 Breast Cancer":

    st.markdown(
        """
        <div class="card">

        <div class="section-title">
        Breast Cancer Risk Analysis
        </div>

        <div class="section-description">
        Enter the 30 cellular measurements used by the
        trained RBF SVM classification model.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    if not breast_loaded:

        st.error(
            "Breast cancer model could not be loaded."
        )

        st.code(
            breast_error
        )

        st.stop()


    # ========================================================
    # SAMPLE DATA
    # ========================================================

    sample_values = [

        17.99,
        10.38,
        122.8,
        1001.0,
        0.1184,
        0.2776,
        0.3001,
        0.1471,
        0.2419,
        0.07871,

        0.9053,
        1.095,
        8.589,
        153.4,
        0.006399,
        0.04904,
        0.05373,
        0.01587,
        0.03003,
        0.006193,

        25.38,
        17.33,
        184.6,
        2019.0,
        0.1622,
        0.6656,
        0.7119,
        0.2654,
        0.4601,
        0.1189

    ]


    # ========================================================
    # FORM
    # ========================================================

    with st.form(
        "breast_prediction_form"
    ):

        st.subheader(
            "Mean Features"
        )

        mean_features = breast_features[:10]

        mean_values = []


        col1, col2 = st.columns(2)


        for i, feature in enumerate(
            mean_features
        ):

            with (
                col1
                if i % 2 == 0
                else col2
            ):

                value = st.number_input(
                    feature.replace(
                        "_",
                        " "
                    ).title(),

                    value=float(
                        sample_values[i]
                    ),

                    format="%.6f",

                    key=f"breast_{feature}"
                )

                mean_values.append(
                    value
                )


        st.subheader(
            "Standard Error Features"
        )

        se_features = breast_features[10:20]

        se_values = []


        col1, col2 = st.columns(2)


        for i, feature in enumerate(
            se_features
        ):

            with (
                col1
                if i % 2 == 0
                else col2
            ):

                value = st.number_input(
                    feature.replace(
                        "_",
                        " "
                    ).title(),

                    value=float(
                        sample_values[10 + i]
                    ),

                    format="%.6f",

                    key=f"breast_{feature}"
                )

                se_values.append(
                    value
                )


        st.subheader(
            "Worst Features"
        )

        worst_features = breast_features[20:30]

        worst_values = []


        col1, col2 = st.columns(2)


        for i, feature in enumerate(
            worst_features
        ):

            with (
                col1
                if i % 2 == 0
                else col2
            ):

                value = st.number_input(
                    feature.replace(
                        "_",
                        " "
                    ).title(),

                    value=float(
                        sample_values[20 + i]
                    ),

                    format="%.6f",

                    key=f"breast_{feature}"
                )

                worst_values.append(
                    value
                )


        st.divider()


        submitted = st.form_submit_button(
            "🔍 Analyze Breast Cancer Risk",
            use_container_width=True
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    if submitted:

        values = (
            mean_values
            +
            se_values
            +
            worst_values
        )


        input_df = pd.DataFrame(
            [values],
            columns=breast_features
        )


        # Scale
        scaled_data = breast_scaler.transform(
            input_df
        )


        # Probability
        probability = breast_model.predict_proba(
            scaled_data
        )[0][1]


        prediction = (
            "Malignant"
            if probability >= breast_threshold
            else "Benign"
        )


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )


        if prediction == "Malignant":

            st.error(
                "⚠️ Malignant Classification"
            )

        else:

            st.success(
                "✓ Benign Classification"
            )


        st.metric(
            "Malignant Probability",
            f"{probability * 100:.2f}%"
        )


        st.progress(
            min(
                float(probability),
                1.0
            )
        )


        col1, col2 = st.columns(2)


        with col1:

            st.write(
                "**Model**"
            )

            st.write(
                "RBF SVM"
            )


        with col2:

            st.write(
                "**Decision Threshold**"
            )

            st.write(
                f"{breast_threshold:.2f}"
            )


        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


# ============================================================
# SKIN CANCER PAGE
# ============================================================

else:

    st.markdown(
        """
        <div class="card">

        <div class="section-title">
        Skin Cancer Detection
        </div>

        <div class="section-description">
        Upload a dermoscopic image for CNN classification
        using the fine-tuned ResNet18 model.
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    if not skin_loaded:

        st.error(
            "Skin ResNet18 model could not be loaded."
        )

        st.code(
            skin_error
        )

        st.stop()


    # ========================================================
    # UPLOAD
    # ========================================================

    uploaded_file = st.file_uploader(
        "Upload Skin Lesion Image",

        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )


    if uploaded_file is not None:

        original_image = Image.open(
            uploaded_file
        ).convert(
            "RGB"
        )


        st.subheader(
            "Selected Image"
        )


        col1, col2 = st.columns(
            [1, 1]
        )


        with col1:

            st.image(
                original_image,
                caption="Uploaded Dermoscopic Image",
                use_container_width=True
            )


        with col2:

            st.markdown(
                """
                <div class="info-box">

                <b>Image ready for analysis</b>

                <br><br>

                The image will be processed using
                the fine-tuned ResNet18 CNN.

                </div>
                """,
                unsafe_allow_html=True
            )


        # ====================================================
        # ANALYZE
        # ====================================================

        if st.button(
            "🔬 Analyze Skin Image",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing image..."
            ):

                # ------------------------------------------------
                # Transform image
                # ------------------------------------------------

                image_tensor = skin_transform(
                    original_image
                ).unsqueeze(
                    0
                )


                # ------------------------------------------------
                # Prediction
                # ------------------------------------------------

                with torch.no_grad():

                    output = skin_model(
                        image_tensor.to(
                            skin_device
                        )
                    )


                    probabilities = torch.softmax(
                        output,
                        dim=1
                    )


                    prediction_index = (
                        torch.argmax(
                            probabilities,
                            dim=1
                        ).item()
                    )


                    # Class 1 = malignant
                    malignant_probability = (
                        probabilities[
                            0,
                            1
                        ].item()
                    )


                if prediction_index == 1:

                    prediction = "Malignant"

                else:

                    prediction = "Benign"


            # ====================================================
            # RESULT
            # ====================================================

            st.divider()

            st.subheader(
                "AI Analysis Result"
            )


            if prediction == "Malignant":

                st.error(
                    "⚠️ Malignant Classification"
                )

            else:

                st.success(
                    "✓ Benign Classification"
                )


            st.metric(
                "Malignant Probability",
                f"{malignant_probability * 100:.2f}%"
            )


            st.progress(
                min(
                    malignant_probability,
                    1.0
                )
            )


            col1, col2, col3 = st.columns(3)


            with col1:

                st.write(
                    "**Model**"
                )

                st.write(
                    "Fine-Tuned ResNet18"
                )


            with col2:

                st.write(
                    "**Prediction**"
                )

                st.write(
                    prediction
                )


            with col3:

                st.write(
                    "**Device**"
                )

                st.write(
                    str(
                        skin_device
                    )
                )


            # ====================================================
            # GRAD-CAM
            # ====================================================

            st.divider()

            st.subheader(
                "🧠 Grad-CAM Model Interpretation"
            )


            st.write(
                "The heatmap highlights image regions "
                "that influenced the CNN prediction."
            )


            # Need gradients
            skin_model.eval()


            image_tensor.requires_grad = True


            cam, cam_prediction = (
                generate_gradcam(
                    skin_model,
                    image_tensor,
                    skin_device
                )
            )


            fig = create_gradcam_overlay(
                original_image,
                cam
            )


            st.pyplot(
                fig,
                clear_figure=True
            )


            st.caption(
                "Grad-CAM is an interpretability visualization "
                "and does not represent a medical diagnosis."
            )


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">

        <div class="disclaimer-title">
            ⚠️ Important Medical Disclaimer
        </div>

        <div>
            OncoVision AI is a research and educational project.
            Its predictions are not medical diagnoses and should
            not be used to diagnose, treat, or rule out cancer.
            Always consult a qualified healthcare professional
            for medical decisions.
        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <br>

    <center>

    <b>OncoVision AI</b>

    <br>

    AI-powered multi-cancer detection research system

    <br><br>

    Developed by Vinay Sharma

    </center>
    """,
    unsafe_allow_html=True
)