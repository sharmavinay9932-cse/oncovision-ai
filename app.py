# ============================================================
# ONCOVISION AI
# Multi-Cancer Detection System
# Developed by Vinay Sharma
# ============================================================

# ------------------------------------------------------------
# IMPORTANT: OpenMP fix must come before importing torch
# ------------------------------------------------------------
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# ------------------------------------------------------------
# IMPORTS
# ------------------------------------------------------------
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


# ============================================================
# PAGE CONFIG
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------
# Breast Cancer
# -------------------------

BREAST_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "breast"
)

BREAST_SVM_PATH = os.path.join(
    BREAST_MODEL_DIR,
    "svm_model.pkl"
)

BREAST_SCALER_PATH = os.path.join(
    BREAST_MODEL_DIR,
    "scaler.pkl"
)

BREAST_CONFIG_PATH = os.path.join(
    BREAST_MODEL_DIR,
    "config.pkl"
)

# -------------------------
# Skin Cancer
# -------------------------

SKIN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "skin",
    "best_resnet18_finetuned.pth"
)

# -------------------------
# Lung Cancer
# -------------------------

LUNG_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "lung",
    "best_resnet18_lung_finetuned.pth"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main title */
    .main-title {
        font-size: 48px;
        font-weight: 800;
        color: #2f6df6;
        margin-bottom: 0;
    }

    .subtitle {
        font-size: 20px;
        color: #777;
        margin-top: 0;
        margin-bottom: 30px;
    }

    /* Developer footer */
    .developer {
        text-align: center;
        color: #777;
        font-size: 15px;
        margin-top: 40px;
        padding: 25px;
    }

    /* Disclaimer */
    .warning-box {
        padding: 22px;
        border-radius: 14px;
        background-color: #fff7df;
        border: 1px solid #f0c36d;
        margin-top: 30px;
        color: #5a4500;
    }

    /* About cards */
    .about-card {
        padding: 22px;
        border-radius: 14px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
        min-height: 210px;
    }

    .about-card h3 {
        margin-top: 0;
    }

    /* Status */
    .status-card {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
    }

    /* Section */
    .section-title {
        font-size: 30px;
        font-weight: 700;
        margin-top: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def file_exists(path):
    return os.path.exists(path)


# ============================================================
# BREAST MODEL LOADING
# ============================================================

@st.cache_resource
def load_breast_models():

    if not file_exists(BREAST_SVM_PATH):
        raise FileNotFoundError(
            f"Breast SVM model not found:\n{BREAST_SVM_PATH}"
        )

    if not file_exists(BREAST_SCALER_PATH):
        raise FileNotFoundError(
            f"Breast scaler not found:\n{BREAST_SCALER_PATH}"
        )

    with open(BREAST_SVM_PATH, "rb") as f:
        svm_model = pickle.load(f)

    with open(BREAST_SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    config = None

    if file_exists(BREAST_CONFIG_PATH):

        with open(BREAST_CONFIG_PATH, "rb") as f:
            config = pickle.load(f)

    return svm_model, scaler, config


# ============================================================
# RESNET18 ARCHITECTURE
# ============================================================

def create_resnet18(num_classes):

    model = resnet18(weights=None)

    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(
            model.fc.in_features,
            num_classes
        )
    )

    return model


# ============================================================
# CLEAN CHECKPOINT
# ============================================================

def clean_state_dict(state_dict):

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        # Remove DataParallel prefix
        if new_key.startswith("module."):
            new_key = new_key[7:]

        cleaned[new_key] = value

    return cleaned


# ============================================================
# UNIVERSAL RESNET CHECKPOINT LOADER
# ============================================================

def load_resnet_checkpoint(model, checkpoint_path):

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Detect checkpoint format
    # --------------------------------------------------------

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint.state_dict()

    # --------------------------------------------------------
    # Clean keys
    # --------------------------------------------------------

    state_dict = clean_state_dict(state_dict)

    # --------------------------------------------------------
    # FIX CLASSIFIER MISMATCH
    #
    # Model:
    # fc.0 = Dropout
    # fc.1 = Linear
    #
    # Some checkpoints:
    # fc.weight
    # fc.bias
    #
    # Convert old format -> current format
    # --------------------------------------------------------

    if (
        "fc.weight" in state_dict
        and "fc.bias" in state_dict
    ):

        state_dict["fc.1.weight"] = state_dict.pop(
            "fc.weight"
        )

        state_dict["fc.1.bias"] = state_dict.pop(
            "fc.bias"
        )

    # --------------------------------------------------------
    # Reverse compatibility
    # If somehow checkpoint has fc.1 but model has fc
    # --------------------------------------------------------

    if (
        "fc.1.weight" in state_dict
        and "fc.1.bias" in state_dict
        and
        "fc.weight" not in state_dict
    ):

        # Current model already expects fc.1
        pass

    # --------------------------------------------------------
    # Load weights
    # --------------------------------------------------------

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False
    )

    # --------------------------------------------------------
    # Important missing parameters
    # --------------------------------------------------------

    important_missing = [
        key
        for key in missing
        if not key.startswith("fc.")
    ]

    if important_missing:

        raise RuntimeError(
            "Important model parameters are missing:\n"
            + "\n".join(important_missing)
        )

    return model


# ============================================================
# SKIN MODEL
# ============================================================

@st.cache_resource
def load_skin_model():

    if not file_exists(SKIN_MODEL_PATH):

        raise FileNotFoundError(
            f"Skin model not found:\n{SKIN_MODEL_PATH}"
        )

    model = create_resnet18(2)

    model = load_resnet_checkpoint(
        model,
        SKIN_MODEL_PATH
    )

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# LUNG MODEL
# ============================================================

@st.cache_resource
def load_lung_model():

    if not file_exists(LUNG_MODEL_PATH):

        raise FileNotFoundError(
            f"Lung model not found:\n{LUNG_MODEL_PATH}"
        )

    model = create_resnet18(4)

    model = load_resnet_checkpoint(
        model,
        LUNG_MODEL_PATH
    )

    model = model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

IMAGE_TRANSFORM = transforms.Compose([

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
])


# ============================================================
# GRAD-CAM CLASS
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = (
            target_layer.register_forward_hook(
                self.save_activation
            )
        )

        self.backward_hook = (
            target_layer.register_full_backward_hook(
                self.save_gradient
            )
        )

    def save_activation(
        self,
        module,
        input,
        output
    ):

        self.activations = output

    def save_gradient(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    def generate(
        self,
        image_tensor,
        target_class
    ):

        self.model.zero_grad()

        output = self.model(
            image_tensor
        )

        score = output[
            0,
            target_class
        ]

        score.backward()

        activations = self.activations[0]

        gradients = self.gradients[0]

        weights = gradients.mean(
            dim=(1, 2)
        )

        cam = torch.zeros(
            activations.shape[1:],
            device=activations.device
        )

        for i, weight in enumerate(weights):

            cam += (
                weight *
                activations[i]
            )

        cam = torch.relu(cam)

        cam -= cam.min()

        if cam.max() > 0:

            cam /= cam.max()

        return cam.detach().cpu().numpy()


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    image,
    predicted_class
):

    input_tensor = (
        IMAGE_TRANSFORM(image)
        .unsqueeze(0)
        .to(DEVICE)
    )

    # ResNet18 final convolutional layer
    target_layer = model.layer4[-1].conv2

    gradcam = GradCAM(
        model,
        target_layer
    )

    try:

        cam = gradcam.generate(
            input_tensor,
            predicted_class
        )

    finally:

        gradcam.forward_hook.remove()
        gradcam.backward_hook.remove()

    # Resize CAM to original image
    cam_image = Image.fromarray(
        np.uint8(cam * 255)
    )

    cam_image = cam_image.resize(
        image.size
    )

    cam = np.array(
        cam_image
    ) / 255.0

    return cam


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧬 OncoVision AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered Multi-Cancer Detection System'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🧬 OncoVision AI"
)

st.sidebar.markdown(
    "**Cancer Detection Platform**"
)

st.sidebar.markdown("---")


# ============================================================
# NAVIGATION
# ============================================================

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📖 About / Description"
    ]
)


# ============================================================
# MODEL STATUS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader(
    "Model Status"
)


breast_exists = (
    file_exists(BREAST_SVM_PATH)
    and
    file_exists(BREAST_SCALER_PATH)
)

skin_exists = file_exists(
    SKIN_MODEL_PATH
)

lung_exists = file_exists(
    LUNG_MODEL_PATH
)


if breast_exists:

    st.sidebar.success(
        "Breast SVM: Available"
    )

else:

    st.sidebar.error(
        "Breast SVM: Missing"
    )


if skin_exists:

    st.sidebar.success(
        "Skin ResNet18: Available"
    )

else:

    st.sidebar.error(
        "Skin ResNet18: Missing"
    )


if lung_exists:

    st.sidebar.success(
        "Lung ResNet18: Available"
    )

else:

    st.sidebar.error(
        "Lung ResNet18: Missing"
    )


st.sidebar.markdown("---")

st.sidebar.caption(
    f"Device: {DEVICE}"
)

st.sidebar.caption(
    "Developed by Vinay Sharma"
)


# ============================================================
# ABOUT / DESCRIPTION PAGE
# ============================================================

if page == "📖 About / Description":

    st.header(
        "📖 About OncoVision AI"
    )

    st.write(
        """
        **OncoVision AI** is an AI-powered multi-cancer
        detection system developed as a research and
        educational project.

        The application demonstrates how machine learning,
        deep learning and computer vision can be used to
        classify cancer-related datasets and medical images.
        """
    )

    st.markdown("---")

    # --------------------------------------------------------
    # PROJECT OBJECTIVE
    # --------------------------------------------------------

    st.header(
        "🎯 Project Objective"
    )

    st.write(
        """
        The main objective of OncoVision AI is to create a
        single interactive platform that demonstrates
        multiple machine-learning approaches for cancer
        classification.
        """
    )

    # --------------------------------------------------------
    # THREE MODELS
    # --------------------------------------------------------

    st.header(
        "🧠 Cancer Detection Modules"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="about-card">

            <h3>🩺 Breast Cancer</h3>

            <b>Model:</b> RBF SVM

            <br><br>

            Uses 30 cellular measurements including:

            <br><br>

            • Radius<br>
            • Texture<br>
            • Perimeter<br>
            • Area<br>
            • Smoothness<br>
            • Compactness<br>
            • Concavity<br>
            • Concave Points<br>
            • Symmetry<br>
            • Fractal Dimension

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="about-card">

            <h3>🔬 Skin Cancer</h3>

            <b>Model:</b> Fine-tuned ResNet18

            <br><br>

            Analyzes dermoscopic skin
            lesion images.

            <br><br>

            <b>Classes:</b>

            <br>

            • Benign<br>
            • Malignant

            <br><br>

            Includes Grad-CAM explainability.

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="about-card">

            <h3>🫁 Lung Cancer</h3>

            <b>Model:</b> Fine-tuned ResNet18

            <br><br>

            Analyzes lung images.

            <br><br>

            <b>Classes:</b>

            <br>

            • Normal<br>
            • Adenocarcinoma<br>
            • Large Cell Carcinoma<br>
            • Squamous Cell Carcinoma

            <br><br>

            Includes Grad-CAM explainability.

            </div>
            """,
            unsafe_allow_html=True
        )

    # --------------------------------------------------------
    # WORKFLOW
    # --------------------------------------------------------

    st.header(
        "⚙️ Machine Learning Workflow"
    )

    st.markdown(
        """
        ### 🩺 Breast Cancer

        **Dataset → Data Cleaning → Preprocessing → Scaling → RBF SVM → Prediction**

        ### 🔬 Skin Cancer

        **Images → Resize → Normalize → ResNet18 → Fine-Tuning → Prediction → Grad-CAM**

        ### 🫁 Lung Cancer

        **Images → Resize → Normalize → ResNet18 → Fine-Tuning → Prediction → Grad-CAM**
        """
    )

    # --------------------------------------------------------
    # GRAD CAM
    # --------------------------------------------------------

    st.header(
        "🔥 Explainable AI — Grad-CAM"
    )

    st.write(
        """
        OncoVision AI uses Grad-CAM for the image-based
        ResNet18 models.

        Grad-CAM generates a heatmap showing image regions
        that contributed to the model's prediction.
        """
    )

    st.info(
        """
        The Grad-CAM visualization is intended for
        educational and model-interpretability purposes.
        It is not a medically validated localization tool.
        """
    )

    # --------------------------------------------------------
    # TECHNOLOGIES
    # --------------------------------------------------------

    st.header(
        "🛠️ Technologies Used"
    )

    tech1, tech2 = st.columns(2)

    with tech1:

        st.markdown(
            """
            **Machine Learning**

            - Python
            - NumPy
            - Pandas
            - Scikit-learn
            - PyTorch
            - Torchvision
            - RBF SVM
            - ResNet18
            """
        )

    with tech2:

        st.markdown(
            """
            **Application & Visualization**

            - Streamlit
            - Matplotlib
            - PIL
            - Grad-CAM
            - Pickle
            """
        )

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.header(
        "✨ Key Features"
    )

    features = [
        "Multi-cancer detection",
        "Breast cancer numerical prediction",
        "Skin lesion image classification",
        "Lung cancer image classification",
        "Class probability display",
        "Confidence score",
        "Grad-CAM explainability",
        "Interactive Streamlit interface",
        "Local model inference"
    ]

    for feature in features:

        st.markdown(
            f"✅ {feature}"
        )

    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    st.header(
        "⚠️ Limitations"
    )

    st.write(
        """
        Model performance depends on the datasets, image
        quality, preprocessing techniques, training procedure
        and model architecture.

        This application has been created for research,
        learning and educational demonstration.
        """
    )

    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="warning-box">

        <h3>⚠️ Important Medical Disclaimer</h3>

        OncoVision AI is a research and educational project.
        Its predictions are not medical diagnoses and should
        not be used to diagnose, treat, or rule out cancer.

        <br><br>

        Always consult a qualified healthcare professional
        for medical decisions.

        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # DEVELOPER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="developer">

        🧬 <b>OncoVision AI</b>

        <br><br>

        Developed by <b>Vinay Sharma</b>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HOME PAGE
# ============================================================

else:

    # --------------------------------------------------------
    # CANCER SELECTION
    # --------------------------------------------------------

    st.sidebar.markdown(
        "### Select Cancer Type"
    )

    cancer_type = st.sidebar.radio(
        " ",
        [
            "🩺 Breast Cancer",
            "🔬 Skin Cancer",
            "🫁 Lung Cancer"
        ]
    )

    # ========================================================
    # BREAST CANCER
    # ========================================================

    if cancer_type == "🩺 Breast Cancer":

        st.header(
            "🩺 Breast Cancer Detection"
        )

        st.write(
            "Enter the 30 cellular measurements "
            "used by the RBF SVM model."
        )

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------

        feature_names = [

            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave_points_mean",
            "symmetry_mean",
            "fractal_dimension_mean",

            "radius_se",
            "texture_se",
            "perimeter_se",
            "area_se",
            "smoothness_se",
            "compactness_se",
            "concavity_se",
            "concave_points_se",
            "symmetry_se",
            "fractal_dimension_se",

            "radius_worst",
            "texture_worst",
            "perimeter_worst",
            "area_worst",
            "smoothness_worst",
            "compactness_worst",
            "concavity_worst",
            "concave_points_worst",
            "symmetry_worst",
            "fractal_dimension_worst"
        ]

        # ----------------------------------------------------
        # SAMPLE DATA
        # ----------------------------------------------------

        sample_values = [

            17.99,
            10.38,
            122.8,
            1001,
            0.1184,
            0.2776,
            0.3001,
            0.1471,
            0.2419,
            0.07871,

            1.095,
            0.9053,
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
            2019,
            0.1622,
            0.6656,
            0.7119,
            0.2654,
            0.4601,
            0.1189
        ]

        if "breast_values" not in st.session_state:

            st.session_state.breast_values = (
                sample_values.copy()
            )

        # ----------------------------------------------------
        # LOAD SAMPLE
        # ----------------------------------------------------

        if st.button(
            "📥 Load Sample Data",
            use_container_width=True
        ):

            st.session_state.breast_values = (
                sample_values.copy()
            )

            st.rerun()

        values = []

        # ----------------------------------------------------
        # MEAN FEATURES
        # ----------------------------------------------------

        st.subheader(
            "Mean Features"
        )

        cols = st.columns(2)

        for i in range(10):

            with cols[i % 2]:

                value = st.number_input(

                    feature_names[i]
                    .replace("_", " ")
                    .title(),

                    value=float(
                        st.session_state
                        .breast_values[i]
                    ),

                    format="%.6f",

                    key=f"breast_{i}"
                )

                values.append(value)

        # ----------------------------------------------------
        # STANDARD ERROR
        # ----------------------------------------------------

        st.subheader(
            "Standard Error Features"
        )

        cols = st.columns(2)

        for i in range(10, 20):

            with cols[(i - 10) % 2]:

                value = st.number_input(

                    feature_names[i]
                    .replace("_", " ")
                    .title(),

                    value=float(
                        st.session_state
                        .breast_values[i]
                    ),

                    format="%.6f",

                    key=f"breast_{i}"
                )

                values.append(value)

        # ----------------------------------------------------
        # WORST
        # ----------------------------------------------------

        st.subheader(
            "Worst Features"
        )

        cols = st.columns(2)

        for i in range(20, 30):

            with cols[(i - 20) % 2]:

                value = st.number_input(

                    feature_names[i]
                    .replace("_", " ")
                    .title(),

                    value=float(
                        st.session_state
                        .breast_values[i]
                    ),

                    format="%.6f",

                    key=f"breast_{i}"
                )

                values.append(value)

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        if st.button(
            "🔍 Predict Breast Cancer",
            type="primary",
            use_container_width=True
        ):

            try:

                svm_model, scaler, config = (
                    load_breast_models()
                )

                X = np.array(
                    values,
                    dtype=float
                ).reshape(
                    1,
                    -1
                )

                X_scaled = scaler.transform(
                    X
                )

                prediction = int(
                    svm_model.predict(
                        X_scaled
                    )[0]
                )

                # ------------------------------------------------
                # PROBABILITY
                # ------------------------------------------------

                probability = None

                if hasattr(
                    svm_model,
                    "predict_proba"
                ):

                    probabilities = (
                        svm_model.predict_proba(
                            X_scaled
                        )[0]
                    )

                    classes = svm_model.classes_

                    if prediction in classes:

                        pred_index = list(
                            classes
                        ).index(prediction)

                        probability = (
                            probabilities[
                                pred_index
                            ]
                        )

                # ------------------------------------------------
                # RESULT
                # ------------------------------------------------

                st.subheader(
                    "Prediction"
                )

                # Preserve the mapping used by your current model
                if prediction == 0:

                    st.error(
                        "⚠️ Prediction: Malignant"
                    )

                else:

                    st.success(
                        "✅ Prediction: Benign"
                    )

                if probability is not None:

                    st.metric(
                        "Model Confidence",
                        f"{probability:.2%}"
                    )

            except Exception as e:

                st.error(
                    f"Breast model error: {e}"
                )

    # ========================================================
    # SKIN CANCER
    # ========================================================

    elif cancer_type == "🔬 Skin Cancer":

        st.header(
            "🔬 Skin Cancer Detection"
        )

        st.write(
            """
            Upload a dermoscopic skin lesion image
            for ResNet18 classification and Grad-CAM
            analysis.
            """
        )

        if not skin_exists:

            st.error(
                "Skin ResNet18 model not found."
            )

        else:

            uploaded_file = st.file_uploader(

                "Upload Skin Lesion Image",

                type=[
                    "jpg",
                    "jpeg",
                    "png",
                    "webp"
                ],

                key="skin_uploader"
            )

            if uploaded_file:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                st.subheader(
                    "Selected Image"
                )

                st.image(
                    image,
                    use_container_width=True
                )

                if st.button(
                    "🔍 Analyze Skin Image",
                    type="primary",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing skin image..."
                    ):

                        try:

                            model = load_skin_model()

                            input_tensor = (
                                IMAGE_TRANSFORM(
                                    image
                                )
                                .unsqueeze(0)
                                .to(DEVICE)
                            )

                            # ------------------------------------
                            # Prediction
                            # ------------------------------------

                            with torch.no_grad():

                                output = model(
                                    input_tensor
                                )

                                probabilities = (
                                    torch.softmax(
                                        output,
                                        dim=1
                                    )
                                )

                                predicted_class = (
                                    torch.argmax(
                                        probabilities,
                                        dim=1
                                    ).item()
                                )

                                confidence = (
                                    probabilities[
                                        0,
                                        predicted_class
                                    ].item()
                                )

                            skin_classes = [
                                "Benign",
                                "Malignant"
                            ]

                            predicted_name = (
                                skin_classes[
                                    predicted_class
                                ]
                            )

                            # ------------------------------------
                            # Result
                            # ------------------------------------

                            st.subheader(
                                "Prediction"
                            )

                            if predicted_class == 1:

                                st.error(
                                    f"⚠️ {predicted_name}"
                                )

                            else:

                                st.success(
                                    f"✅ {predicted_name}"
                                )

                            st.metric(
                                "Confidence",
                                f"{confidence:.2%}"
                            )

                            # ------------------------------------
                            # Probabilities
                            # ------------------------------------

                            st.subheader(
                                "Class Probabilities"
                            )

                            probability_df = pd.DataFrame({

                                "Class":
                                    skin_classes,

                                "Probability": [

                                    float(p)

                                    for p in
                                    probabilities[
                                        0
                                    ].cpu().numpy()

                                ]

                            })

                            probability_df[
                                "Probability"
                            ] = (
                                probability_df[
                                    "Probability"
                                ].map(
                                    lambda x:
                                    f"{x:.2%}"
                                )
                            )

                            st.dataframe(
                                probability_df,
                                use_container_width=True,
                                hide_index=True
                            )

                            # ------------------------------------
                            # Grad-CAM
                            # ------------------------------------

                            st.subheader(
                                "🔥 Grad-CAM Explainability"
                            )

                            with st.spinner(
                                "Generating Grad-CAM..."
                            ):

                                cam = generate_gradcam(
                                    model,
                                    image,
                                    predicted_class
                                )

                            col1, col2, col3 = (
                                st.columns(3)
                            )

                            # Original
                            with col1:

                                st.image(
                                    image,
                                    caption="Original Image",
                                    use_container_width=True
                                )

                            # Heatmap
                            with col2:

                                fig, ax = plt.subplots()

                                ax.imshow(
                                    cam,
                                    cmap="jet"
                                )

                                ax.axis("off")

                                ax.set_title(
                                    "Grad-CAM"
                                )

                                st.pyplot(
                                    fig
                                )

                                plt.close(fig)

                            # Overlay
                            with col3:

                                fig, ax = plt.subplots()

                                ax.imshow(
                                    image
                                )

                                ax.imshow(
                                    cam,
                                    cmap="jet",
                                    alpha=0.45
                                )

                                ax.axis("off")

                                ax.set_title(
                                    "Grad-CAM Overlay"
                                )

                                st.pyplot(
                                    fig
                                )

                                plt.close(fig)

                        except Exception as e:

                            st.error(
                                f"Skin prediction error: {e}"
                            )


    # ========================================================
    # LUNG CANCER
    # ========================================================

    elif cancer_type == "🫁 Lung Cancer":

        st.header(
            "🫁 Lung Cancer Detection"
        )

        st.write(
            """
            Upload a lung image for four-class
            ResNet18 classification and Grad-CAM analysis.
            """
        )

        if not lung_exists:

            st.error(
                "Lung ResNet18 model not found."
            )

        else:

            uploaded_file = st.file_uploader(

                "Upload Lung Image",

                type=[
                    "jpg",
                    "jpeg",
                    "png",
                    "webp"
                ],

                key="lung_uploader"
            )

            if uploaded_file:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                st.subheader(
                    "Selected Image"
                )

                st.image(
                    image,
                    use_container_width=True
                )

                if st.button(
                    "🔍 Analyze Lung Image",
                    type="primary",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Analyzing lung image..."
                    ):

                        try:

                            model = load_lung_model()

                            input_tensor = (
                                IMAGE_TRANSFORM(
                                    image
                                )
                                .unsqueeze(0)
                                .to(DEVICE)
                            )

                            # ------------------------------------
                            # Prediction
                            # ------------------------------------

                            with torch.no_grad():

                                output = model(
                                    input_tensor
                                )

                                probabilities = (
                                    torch.softmax(
                                        output,
                                        dim=1
                                    )
                                )

                                predicted_class = (
                                    torch.argmax(
                                        probabilities,
                                        dim=1
                                    ).item()
                                )

                                confidence = (
                                    probabilities[
                                        0,
                                        predicted_class
                                    ].item()
                                )

                            # IMPORTANT:
                            # These match the class order you
                            # used during your lung evaluation.
                            lung_classes = [

                                "Normal",

                                "Adenocarcinoma",

                                "Large Cell Carcinoma",

                                "Squamous Cell Carcinoma"

                            ]

                            predicted_name = (
                                lung_classes[
                                    predicted_class
                                ]
                            )

                            # ------------------------------------
                            # Prediction result
                            # ------------------------------------

                            st.subheader(
                                "Prediction"
                            )

                            if predicted_class == 0:

                                st.success(
                                    f"✅ {predicted_name}"
                                )

                            else:

                                st.warning(
                                    f"⚠️ {predicted_name}"
                                )

                            st.metric(
                                "Confidence",
                                f"{confidence:.2%}"
                            )

                            # ------------------------------------
                            # Class probabilities
                            # ------------------------------------

                            st.subheader(
                                "Class Probabilities"
                            )

                            probability_df = pd.DataFrame({

                                "Class":
                                    lung_classes,

                                "Probability": [

                                    float(p)

                                    for p in
                                    probabilities[
                                        0
                                    ].cpu().numpy()

                                ]

                            })

                            probability_df[
                                "Probability"
                            ] = (
                                probability_df[
                                    "Probability"
                                ].map(
                                    lambda x:
                                    f"{x:.2%}"
                                )
                            )

                            st.dataframe(
                                probability_df,
                                use_container_width=True,
                                hide_index=True
                            )

                            # ------------------------------------
                            # Grad-CAM
                            # ------------------------------------

                            st.subheader(
                                "🔥 Grad-CAM Explainability"
                            )

                            with st.spinner(
                                "Generating Grad-CAM..."
                            ):

                                cam = generate_gradcam(
                                    model,
                                    image,
                                    predicted_class
                                )

                            col1, col2, col3 = (
                                st.columns(3)
                            )

                            # Original
                            with col1:

                                st.image(
                                    image,
                                    caption="Original Image",
                                    use_container_width=True
                                )

                            # Heatmap
                            with col2:

                                fig, ax = plt.subplots()

                                ax.imshow(
                                    cam,
                                    cmap="jet"
                                )

                                ax.axis("off")

                                ax.set_title(
                                    "Grad-CAM"
                                )

                                st.pyplot(
                                    fig
                                )

                                plt.close(fig)

                            # Overlay
                            with col3:

                                fig, ax = plt.subplots()

                                ax.imshow(
                                    image
                                )

                                ax.imshow(
                                    cam,
                                    cmap="jet",
                                    alpha=0.45
                                )

                                ax.axis("off")

                                ax.set_title(
                                    "Grad-CAM Overlay"
                                )

                                st.pyplot(
                                    fig
                                )

                                plt.close(fig)

                        except Exception as e:

                            st.error(
                                f"Lung prediction error: {e}"
                            )


# ============================================================
# MEDICAL DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="warning-box">

    <h3>⚠️ Important Medical Disclaimer</h3>

    OncoVision AI is a research and educational project.
    Its predictions are not medical diagnoses and should
    not be used to diagnose, treat, or rule out cancer.

    <br><br>

    Always consult a qualified healthcare professional
    for medical decisions.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="developer">

        🧬 <b>OncoVision AI</b>

        <br><br>

        Developed by <b>Vinay Sharma</b>

        <br><br>

        Multi-Cancer Detection • Machine Learning •
        Computer Vision • Explainable AI

    </div>
    """,
    unsafe_allow_html=True
)