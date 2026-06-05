import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Fruit Freshness Detector 🍎",
    page_icon="🍏",
    layout="centered"
)

# =========================
# BACKGROUND + UI
# =========================
def set_bg(image_url):

    st.markdown(
        f"""
        <style>

        /* MAIN BACKGROUND */
        .stApp {{
            background: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        /* LIGHT OVERLAY */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;

            background: rgba(255,255,255,0.25);

            z-index: 0;
        }}

        /* MAIN CONTAINER */
        .block-container {{
            position: relative;
            z-index: 1;

            background: rgba(0,0,0,0.55);

            backdrop-filter: blur(8px);

            border-radius: 25px;

            padding: 30px;

            box-shadow: 0 8px 30px rgba(0,0,0,0.4);

            border: 1px solid rgba(255,255,255,0.1);
        }}

        /* TEXT COLOR */
        h1, h2, h3, h4, h5, h6, p, label, div {{
            color: white !important;
        }}

        /* FILE UPLOADER */
        .stFileUploader {{
            background: rgba(0,0,0,0.35);

            padding: 15px;

            border-radius: 15px;
        }}

        /* RESULT BOX */
        .prediction-box {{
            background: rgba(0,0,0,0.45);

            padding: 20px;

            border-radius: 20px;

            margin-top: 20px;

            border: 1px solid rgba(255,255,255,0.1);

            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        }}

        /* BUTTON */
        .stButton>button {{
            background: linear-gradient(
                90deg,
                #ff4d4d,
                #ffcc00
            );

            color: black;

            border-radius: 12px;

            border: none;

            padding: 10px 20px;

            font-weight: bold;

            transition: 0.3s;
        }}

        .stButton>button:hover {{

            transform: scale(1.05);

            background: linear-gradient(
                90deg,
                #00ffcc,
                #00ccff
            );

            color: white;
        }}

        /* PROGRESS BAR */
        .stProgress > div > div > div > div {{
            background: linear-gradient(
                90deg,
                #00ffcc,
                #00ccff
            );
        }}

        </style>
        """,

        unsafe_allow_html=True
    )

# =========================
# BACKGROUND IMAGE
# =========================
set_bg(
    "https://images.unsplash.com/photo-1610832958506-aa56368176cf"
)

# =========================
# TITLE
# =========================
st.markdown("""
<style>

.fruit-title {

    font-size: 58px;

    font-weight: 900;

    text-align: center;

    background: linear-gradient(
        90deg,
        #ff4d4d,
        #ffcc00,
        #00ffcc
    );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

    text-shadow: 0 0 25px rgba(255,255,255,0.3);

    margin-bottom: 10px;

    animation: glow 2s infinite alternate;
}

.subtitle {

    text-align: center;

    color: white;

    font-size: 18px;

    margin-bottom: 30px;
}

@keyframes glow {

    from {
        text-shadow: 0 0 10px rgba(255, 0, 0, 0.3);
    }

    to {
        text-shadow: 0 0 30px rgba(0, 255, 200, 0.8);
    }
}

@media (max-width: 768px) {

    .fruit-title {
        font-size: 36px;
    }
}

</style>

<div class="fruit-title">
🍎 Fruit Freshness Detector
</div>

<div class="subtitle">
Detect Fresh and Rotten Fruits using AI 🍌🍊
</div>

""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(
    "freshness_detector_model.h5"
)

# =========================
# CLASS LABELS
# =========================
classes = [

    'freshapples',
    'freshbanana',
    'freshorange',

    'rottenapples',
    'rottenbanana',
    'rottenorange'
]

# =========================
# FILE UPLOADER
# =========================
uploaded_file = st.file_uploader(

    "📤 Upload Fruit Image",

    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:

    # OPEN IMAGE
    img = Image.open(uploaded_file)

    # SHOW IMAGE
    st.image(

        img,

        caption="📸 Uploaded Fruit Image",

        use_container_width=True
    )

    # RESIZE IMAGE
    img = img.resize((128, 128))

    # IMAGE TO ARRAY
    img_array = image.img_to_array(img)

    # NORMALIZE
    img_array = img_array / 255.0

    # EXPAND DIMENSION
    img_array = np.expand_dims(

        img_array,

        axis=0
    )

    # MODEL PREDICTION
    result = model.predict(img_array)

    prediction = np.argmax(result)

    confidence = np.max(result) * 100

    label = classes[prediction]

    # =========================
    # RESULT SECTION
    # =========================
    st.markdown(
        '<div class="prediction-box">',
        unsafe_allow_html=True
    )

    st.subheader("🔍 Prediction Result")

    st.write(
        f"### 🏷️ Predicted Class: `{label}`"
    )

    st.write(
        f"### 📊 Confidence: `{confidence:.2f}%`"
    )

    # FRESH OR ROTTEN
    if "fresh" in label:

        st.success(
            "✅ This fruit is Fresh and Healthy"
        )

    else:

        st.error(
            "❌ This fruit is Rotten"
        )

    # CONFIDENCE BAR
    st.progress(int(confidence))

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )