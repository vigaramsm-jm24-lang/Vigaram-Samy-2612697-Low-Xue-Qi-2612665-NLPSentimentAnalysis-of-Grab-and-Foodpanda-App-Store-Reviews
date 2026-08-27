"""Small offline verification for the saved models and critical demo examples."""

from pathlib import Path

import joblib

from sentiment_pipeline import preprocess_for_model


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
NB_VECTOR = joblib.load(MODEL_DIR / "member_a_nb_vectorizer.pkl")
NB_MODEL = joblib.load(MODEL_DIR / "member_a_naive_bayes.pkl")
SVM_VECTOR = joblib.load(MODEL_DIR / "member_b_svm_vectorizer.pkl")
SVM_MODEL = joblib.load(MODEL_DIR / "member_b_svm.pkl")

CHECKS = [
    ("Food delicious but I don't like it", "negative"),
    ("Delicious but I don't like it", "negative"),
    ("Delivery fast tapi food tak sedap", "negative"),
    ("Not bad", "positive"),
    ("Tak puas hati", "negative"),
    ("Food 🤮", "negative"),
    ("The order arrived at 7:30 pm.", "neutral"),
    ("Tak sepad", "negative"),  # typo of "tak sedap" -- fuzzy match check
    ("Tak akan datang lagi", "negative"),  # implied negativity, no single negative word
    ("Makanan sik bagus", "negative"),  # Sarawak dialect negation ("sik" = not) + "bagus" (good)
    ("This food is soooo goooood!!!", "positive"),  # elongated words -- destretch check
    ("guuuddd food", "positive"),  # elongated + vowel-swap slang spelling
    ("nooooo this is bad", "negative"),  # elongated negation word must still be detected
    ("baaaaad food", "negative"),  # single-letter word stretched -- must NOT become "baad"
    ("terukkkk service", "negative"),  # Malay single-letter word stretched
    ("very gudddd food", "positive"),  # intensifier + stretched slang spelling
    ("not gud", "negative"),  # spelling correction must happen before negation semantics
    ("food is not guudd", "negative"),
    ("makanan tak gud", "negative"),
    ("delishhh food", "positive"),  # destretch -> "delish" -> must resolve to "delicious"
    ("food damn nice", "positive"),  # regression check: "damn" must NOT fuzzy-match to "namun" (but)
    ("lambat gila delivery", "negative"),  # "gila" intensifier must translate to "very", not stay unknown
    ("nice dish overall", "positive"),  # "dish" stays unchanged; "nice" remains positive
]


def predict(text):
    processed = preprocess_for_model(text)
    return (
        processed,
        str(NB_MODEL.predict(NB_VECTOR.transform([processed]))[0]),
        str(SVM_MODEL.predict(SVM_VECTOR.transform([processed]))[0]),
    )


if __name__ == "__main__":
    failures = []
    for review, expected in CHECKS:
        processed, nb, svm = predict(review)
        print(f"{review!r}\n  expected={expected}, NB={nb}, SVM={svm}\n  {processed}")
        if nb != expected or svm != expected:
            failures.append((review, expected, nb, svm))
    if failures:
        raise SystemExit(f"Critical checks failed: {failures}")
    print(f"All {len(CHECKS)} critical checks passed for both models.")
