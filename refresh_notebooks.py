"""Create and execute the four submission notebooks from the shared workflow."""

from pathlib import Path
import textwrap

import nbformat as nbf
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


common = """
from pathlib import Path
import sys

PROJECT_ROOT = Path.cwd().resolve()
if not (PROJECT_ROOT / 'dataset').exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / 'dataset'
MODEL_DIR = PROJECT_ROOT / 'model'
print('Project root:', PROJECT_ROOT)
"""


notebooks = {
    "1_Text_Label_Preprocessing.ipynb": [
        md("""
        # 1 — Text labels, multilingual patterns, and preprocessing

        This notebook creates sentiment targets from **review text rather than star ratings**.
        VADER and TextBlob provide high-confidence weak labels for real app reviews. A separate,
        team-curated training file adds English, Malay, Manglish, emoji, negation, and contrastive
        sentence patterns. Curated patterns are training-only and never enter the main test set.

        The post-contrast clause after *but/tapi/however/yet* receives explicit focus features.
        Label text and model text are prepared separately, so VADER/TextBlob never receive
        machine-only tokens such as `NOT_good`. Repeated letters, known informal spellings,
        and close variants of the project's sentiment vocabulary are normalised before the
        linguistic phrase rules run a second time; therefore `not gud` becomes `bad`.
        """),
        code(common + """
import pandas as pd
from sentiment_pipeline import normalize_for_labeler, focus_for_labeler, preprocess_for_model
from project_workflow import build_dataset
"""),
        code("""
examples = [
    "Food delicious but I don't like it",
    "Delivery fast tapi food tak sedap",
    "Not bad",
    "Tak puas hati",
    "Not gud",
    "Food is not guudd",
    "Food 🤮",
]
pd.DataFrame({
    'raw_review': examples,
    'label_view': [focus_for_labeler(x) for x in examples],
    'model_features': [preprocess_for_model(x) for x in examples],
})
"""),
        code("""
summary = build_dataset(PROJECT_ROOT)
summary
"""),
        code("""
train = pd.read_csv(DATA_DIR / 'train_split.csv')
test = pd.read_csv(DATA_DIR / 'test_split.csv')
patterns = pd.read_csv(DATA_DIR / 'curated_training_patterns.csv')
assert set(train['normalized_key']).isdisjoint(set(test['normalized_key']))
print('Training label counts:')
print(train['text_sentiment'].value_counts().sort_index())
print()
print('Real held-out test counts:')
print(test['text_sentiment'].value_counts().sort_index())
print()
print('Curated training patterns:', len(patterns))
patterns[['review', 'text_sentiment', 'language_pattern', 'pattern_type']].sample(12, random_state=42)
"""),
        md("""
        Main accuracy is later calculated only on unseen, real app-review rows. The curated
        patterns make important linguistic structures learnable but do not inflate the main
        test score. The separate challenge CSV is a functional diagnostic, not a benchmark.
        """),
    ],
    "2_Member_A_Naive_Bayes.ipynb": [
        md("""
        # 2 — Member A: Complement Naive Bayes

        Complement Naive Bayes is an NB variant designed for sparse text classification.
        Word 1–3 grams and character 2–5 grams represent ordinary vocabulary, phrases,
        spelling variation, negation, and Manglish patterns. Alpha and ComplementNB's norm option
        are selected with five-fold training-only cross-validation. Character features use a 0.5
        weight because training-only CV showed that full weight double-counted correlated evidence
        for NB. The fixed real-review test set is evaluated once.
        """),
        code(common + """
import pandas as pd
from project_workflow import train_naive_bayes
"""),
        code("""
result = train_naive_bayes(PROJECT_ROOT)
pd.DataFrame([result])
"""),
        code("""
report = pd.read_csv(DATA_DIR / 'member_a_nb_classification_report.csv', index_col=0)
matrix = pd.read_csv(DATA_DIR / 'member_a_nb_confusion_matrix.csv', index_col=0)
display(report)
matrix
"""),
    ],
    "3_Member_B_SVM.ipynb": [
        md("""
        # 3 — Member B: Linear Support Vector Machine

        This second NLP solution uses the same fixed train/test rows and the same TF-IDF feature
        families for a fair comparison. Class balancing and C are handled within the SVM setup;
        C is selected by five-fold training-only cross-validation.
        """),
        code(common + """
import pandas as pd
from project_workflow import train_svm
"""),
        code("""
result = train_svm(PROJECT_ROOT)
pd.DataFrame([result])
"""),
        code("""
report = pd.read_csv(DATA_DIR / 'member_b_svm_classification_report.csv', index_col=0)
matrix = pd.read_csv(DATA_DIR / 'member_b_svm_confusion_matrix.csv', index_col=0)
display(report)
matrix
"""),
    ],
    "4_Comparison_and_Demo.ipynb": [
        md("""
        # 4 — Model comparison, challenge test, and demo

        The main table uses the same held-out real-review test set for both models. The multilingual
        challenge contains unseen, team-written sentences that test negation, contrast, Manglish,
        emoji, and neutral factual language. Challenge accuracy is diagnostic and is reported
        separately from held-out test accuracy.
        """),
        code(common + """
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentiment_pipeline import preprocess_for_model, sentiment_emoji
from project_workflow import build_comparison

FIGURE_DIR = PROJECT_ROOT / 'report_figures'
FIGURE_DIR.mkdir(exist_ok=True)
"""),
        code("""
comparison = build_comparison(PROJECT_ROOT)
display_table = comparison.copy()
metric_columns = ['Accuracy', 'Precision_weighted', 'Recall_weighted',
                  'F1_weighted', 'F1_macro', 'Pattern_challenge_accuracy']
for column in metric_columns:
    display_table[column] = (display_table[column] * 100).round(2).astype(str) + '%'
display_table
"""),
        code("""
plot_data = comparison.set_index('Model')[['Accuracy', 'F1_weighted', 'F1_macro']] * 100
ax = plot_data.plot(kind='bar', figsize=(10, 5), ylim=(0, 100), color=sns.color_palette('Set2', 3))
ax.axhline(80, color='crimson', linestyle='--', label='80% target')
ax.set_ylabel('Score (%)')
ax.set_title('Held-out Real-review Test Performance')
ax.tick_params(axis='x', rotation=0)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'model_comparison.png', dpi=180, bbox_inches='tight')
plt.show()
"""),
        code("""
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, filename, title in [
    (axes[0], 'member_a_nb_confusion_matrix.csv', 'Complement Naive Bayes'),
    (axes[1], 'member_b_svm_confusion_matrix.csv', 'Linear SVM'),
]:
    cm = pd.read_csv(DATA_DIR / filename, index_col=0)
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
plt.tight_layout()
plt.savefig(FIGURE_DIR / 'confusion_matrices.png', dpi=180, bbox_inches='tight')
plt.show()
"""),
        code("""
challenge = pd.read_csv(DATA_DIR / 'multilingual_pattern_challenge_results.csv')
print('Naive Bayes challenge:', f"{challenge['nb_correct'].mean():.2%}")
print('Linear SVM challenge:', f"{challenge['svm_correct'].mean():.2%}")
challenge[['review', 'expected_sentiment', 'nb_prediction', 'svm_prediction']]
"""),
        code("""
nb_vectorizer = joblib.load(MODEL_DIR / 'member_a_nb_vectorizer.pkl')
nb_model = joblib.load(MODEL_DIR / 'member_a_naive_bayes.pkl')
svm_vectorizer = joblib.load(MODEL_DIR / 'member_b_svm_vectorizer.pkl')
svm_model = joblib.load(MODEL_DIR / 'member_b_svm.pkl')

def predict_review(review_text, model_name='Linear SVM'):
    processed = preprocess_for_model(review_text)
    if model_name == 'Naive Bayes':
        prediction = str(nb_model.predict(nb_vectorizer.transform([processed]))[0])
    else:
        prediction = str(svm_model.predict(svm_vectorizer.transform([processed]))[0])
    return {'review': review_text, 'model': model_name, 'sentiment': prediction,
            'emoji': sentiment_emoji(prediction), 'processed': processed}

examples = [
    "Food delicious but I don't like it",
    "Delivery fast tapi food tak sedap",
    "Not bad",
    "Tak puas hati",
    "Not gud",
    "Food is not guudd",
    "The order arrived at 7:30 pm.",
]
pd.DataFrame([predict_review(x, model) for x in examples for model in ['Naive Bayes', 'Linear SVM']])
"""),
        md("""
        Report the held-out test accuracy and challenge result separately. The weak-supervision
        accuracy measures agreement with VADER–TextBlob text labels, not human gold labels.
        Star ratings are retained only as audit metadata and are never the prediction target.
        """),
    ],
}


for filename, cells in notebooks.items():
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
    path = MODEL_DIR / filename
    nbf.write(notebook, path)
    with path.open("r", encoding="utf-8") as handle:
        loaded = nbf.read(handle, as_version=4)
    client = NotebookClient(loaded, timeout=1200, kernel_name="python3")
    client.execute(cwd=str(MODEL_DIR))
    nbf.write(loaded, path)
    print("Executed:", filename)
