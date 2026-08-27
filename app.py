from pathlib import Path

import joblib
import streamlit as st

from sentiment_pipeline import preprocess_for_model, sentiment_emoji


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"

st.set_page_config(page_title="Food Delivery Sentiment", page_icon="🍜", layout="centered")
st.markdown(
    """
    <style>
    .sentiment-card {padding: 1.25rem; border-radius: 16px; text-align: center;
                     border: 1px solid #d9d9d9; margin-top: 1rem;}
    .sentiment-emoji {font-size: 4.5rem; line-height: 1.1;}
    .sentiment-label {font-size: 1.7rem; font-weight: 700;}
    [data-testid="stAppDeployButton"] {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🍜 Grab / Foodpanda Sentiment Analyser")
st.caption("English · Malay · Manglish · Emoji | Positive · Neutral · Negative")


@st.cache_resource
def load_models():
    files = {
        "Naive Bayes": (
            MODEL_DIR / "member_a_nb_vectorizer.pkl",
            MODEL_DIR / "member_a_naive_bayes.pkl",
        ),
        "Linear SVM": (
            MODEL_DIR / "member_b_svm_vectorizer.pkl",
            MODEL_DIR / "member_b_svm.pkl",
        ),
    }
    missing = [str(path) for pair in files.values() for path in pair if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing model file(s): " + ", ".join(missing))
    return {
        name: (joblib.load(vectorizer), joblib.load(model))
        for name, (vectorizer, model) in files.items()
    }


try:
    models = load_models()
except Exception as exc:
    st.error("The saved models could not be loaded. Keep app.py beside the model folder.")
    st.exception(exc)
    st.stop()

model_choice = st.radio(
    "Choose a sentiment model",
    ["Linear SVM", "Naive Bayes"],
    horizontal=True,
    help="SVM usually performs better; Naive Bayes is Member A's comparison method.",
)
review = st.text_area(
    "Type a food-delivery review",
    placeholder="Example: Food delicious but I don't like it",
    height=140,
)

if st.button("Analyse review", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Please type a review first.")
    else:
        processed = preprocess_for_model(review)
        vectorizer, model = models[model_choice]
        features = vectorizer.transform([processed])
        prediction = str(model.predict(features)[0])
        icon = sentiment_emoji(prediction)
        if model_choice == "Naive Bayes":
            confidence_text = f"Model probability: {model.predict_proba(features).max():.1%}"
        else:
            confidence_text = f"SVM decision margin: {model.decision_function(features).max():.3f}"

        st.markdown(
            f"""
            <div class="sentiment-card">
                <div class="sentiment-emoji">{icon}</div>
                <div class="sentiment-label">{prediction.title()}</div>
                <div>Model used: {model_choice}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(confidence_text)
        with st.expander("See how the text was processed"):
            st.code(processed, language=None)
            st.write(
                "The text after but/tapi/however/yet receives explicit focus features, "
                "and negated words are marked for the selected machine-learning model."
            )

st.divider()
st.caption(
    "Main accuracy is measured on a fixed held-out set of real app reviews labelled from "
    "review text by VADER–TextBlob agreement. Curated multilingual patterns are training-only; "
    "star ratings are not target labels."
)
