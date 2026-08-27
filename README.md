# Grab/Foodpanda Multilingual Sentiment Analysis

## Current verified result

This project classifies a user review as **negative, neutral, or positive** using two
different NLP solutions. Both results exceed the lecturer's 80% held-out accuracy target:

| Model | Held-out accuracy | Weighted F1 | Multilingual pattern challenge |
|---|---:|---:|---:|
| Complement Naive Bayes | 81.19% | 80.60% | 100.00% (33/33) |
| Linear SVM | 96.53% | 96.53% | 100.00% (33/33) |

The main accuracy is calculated on **4,317 unseen real app reviews**. The 630 curated
English/Malay/Manglish/emoji patterns are added to the training set only. The 33 challenge
sentences are not included in training and are a functional diagnostic, not a general benchmark.

## Important fixes in this version

- Text after `but`, `tapi`, `however`, or `yet` receives explicit focus features. For example,
  `Food delicious but I don't like it` is now negative with both models.
- Multiword Manglish is matched before individual words. `tak puas hati`, `tak sedap`,
  `tak boleh guna`, and similar expressions now work.
- Informal spelling is normalised before phrase rules run again. Therefore `gud/guudd` becomes
  `good`, then `not gud` becomes the explicit negative feature `bad` rather than retaining a
  strong positive `good` token.
- Fuzzy spelling correction is restricted to a safe whitelist of longer sentiment words. This
  prevents normal words such as Malay `saya` from being incorrectly matched to slang `slay`.
- Negation is prepared separately for weak labelling and model features. VADER/TextBlob receive
  readable text; NB/SVM receive features such as `NOT_delicious` where appropriate.
- Emoji are converted to sentiment words. The UI displays 😊, 😐, or 😞 with the result.
- The Streamlit UI lets the user select **Linear SVM** or **Naive Bayes** before typing a review.
- Streamlit's developer/Deploy control is hidden using `client.toolbarMode = "minimal"`.
- Dependency versions are compatible with one another and with the saved scikit-learn models.

## Processing pipeline currently implemented

1. Unicode and apostrophe normalisation.
2. Emoji conversion to readable sentiment terms (`🤮` → `disgusting`).
3. Repeated-letter reduction while preserving real double letters (`goooood` → `good`,
   `baaaaad` → `bad`).
4. English contraction expansion (`don't` → `do not`).
5. Longest-first idiom, Malay, Manglish, internet-slang, and emoji-phrase mapping.
6. Punctuation-aware tokenisation and filler-word removal (`lah`, `lor`, etc.).
7. Exact non-standard spelling mapping (`gud`, `guud`, `guudd`, `gewd`) plus restricted
   high-threshold fuzzy correction for longer safe sentiment words.
8. A second phrase pass after spelling correction, which resolves combinations such as
   `not gud` → `not good` → `bad`.
9. Contrast-clause detection: the clause after `but/tapi/however/yet` receives extra focus.
10. Model-only negation scope marking for remaining negations (`NOT_word`) with punctuation and
    clause boundaries.
11. Word TF-IDF 1–3 grams and character TF-IDF 2–5 grams; the latter helps spelling variation.
12. Complement Naive Bayes and Linear SVM classification, followed by emoji output in the UI.

VADER and TextBlob are used only to create high-confidence weak labels from readable label text.
They do not receive `NOT_` model-feature tokens. Star ratings are audit metadata, not targets.

## Start the user interface

### Easiest method

Double-click:

    run_app.bat

It opens the correct project folder, creates an isolated `.venv`, installs compatible project
packages on the first run, and starts the app. The first run can take a few minutes.

### Manual method

Open PowerShell in the folder containing `app.py`, then run:

    python -m pip install -r requirements.txt
    python -m streamlit run app.py

If your Python is installed in the existing Anaconda folder, use:

    C:\anacoda3\python.exe -m pip install -r requirements.txt
    C:\anacoda3\python.exe -m streamlit run app.py

Open `http://localhost:8501` if the browser does not open automatically.

Do **not** use `python app.py`. That launches Streamlit in bare mode and produces
`missing ScriptRunContext` warnings.

## Notebook order

The notebooks in `model/` are already executed with the current results. To reproduce the full
pipeline, run them in this order:

1. `1_Text_Label_Preprocessing.ipynb`
2. `2_Member_A_Naive_Bayes.ipynb`
3. `3_Member_B_SVM.ipynb`
4. `4_Comparison_and_Demo.ipynb`

Notebook 1 builds the train/test CSV files and curated pattern data. Notebook 2 trains NB,
Notebook 3 trains SVM, and Notebook 4 creates the comparison, challenge results, charts, and demo.

## Key files

- `sentiment_pipeline.py` — shared preprocessing for training, notebooks, verification, and App
- `project_workflow.py` — reproducible data building and model training functions
- `dataset/curated_training_patterns.csv` — 630 training-only multilingual patterns
- `dataset/multilingual_pattern_challenge_results.csv` — unseen diagnostic results
- `dataset/model_comparison.csv` — final held-out and challenge metrics
- `verify_project.py` — offline checks for 23 critical processing and prediction examples
- `run_app.bat` — one-click Windows launcher

## Correct interpretation for the report

The target label does **not** come from star ratings. Real-review targets are high-confidence weak
labels created when VADER and TextBlob agree on the review text after transparent normalisation.
Star ratings remain only as audit metadata. The 630 pattern labels are team-curated linguistic
labels. Therefore, held-out accuracy measures agreement with this text-labelling policy; it must
not be described as human-gold accuracy.

The audit sample in `dataset/text_label_audit_sample.csv` can be completed by team members if an
independent human-labelled benchmark is required.

## Elongated / stretched spelling ("goooood", "baaaaad", "sooo")

Repeated letters (3 or more of the same character in a row) are normalised per
word. Most words collapse fully to a single letter ("baaaaad" -> "bad",
"terukkkk" -> "teruk"), since that's correct for the vast majority of words.
A short list of common words that genuinely have a double letter ("good",
"sweet", "class", "food"...) collapse to 2 instead, so they don't get
incorrectly shortened ("goooood" -> "good", not "god"). Negation words ("no",
"not") get extra handling so "noooooo" still correctly registers as negation.
A short list of common informal spelling variants ("gud", "guud", "guudd") is
also recognised directly -- vowel-swap slang like this can't be safely caught
by the general collapsing rule alone.

## Remaining limitations

- **Sarcasm and indirect humour cannot be solved by this architecture at all**, regardless of
  dataset size. Naive Bayes/SVM count words; they have no representation for tone or the
  expectation-vs-reality gap that makes something sarcastic (e.g. "so fast, only waited 3
  hours"). This would require a context-aware model (a transformer) AND a dedicated
  human-annotated sarcasm dataset, and even then remains an active, unsolved research problem.
- **Regional dialects are only partially covered, not solved.** Sarawak Malay's `sik` negation
  and a small explicit Manglish/Malay vocabulary are recognised. The system does not claim
  comprehensive coverage of Malaysian regional dialects.
- The challenge is deliberately small and diagnostic, so 100% there is not proof of universal
  multilingual accuracy. It also shares templates with the curated training patterns, so it
  demonstrates the model learned the taught patterns, not that it generalises to arbitrary
  novel phrasing — say this precisely in your report rather than implying full generalisation.
- ~~Before submission, add the original download URL and licence for both raw dataset files.~~
  **Resolved.** Dataset sources confirmed:
  - Grab reviews: https://www.kaggle.com/datasets/bwandowando/grab-app-reviews-from-google-store
  - Foodpanda reviews: https://www.kaggle.com/datasets/bwandowando/foodpanda-app-reviews-from-google-store
  - Both scraped from the Google Play Store by the same publisher (bwandowando).
  - **Still needed**: check the licence badge shown on each Kaggle page (usually near the top)
    and note it in your report's Dataset section -- not confirmed here, verify it yourself.
