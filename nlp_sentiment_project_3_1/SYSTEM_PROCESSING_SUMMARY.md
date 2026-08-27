# System Processing Summary

## Processing stages

1. **Unicode normalisation** — standardises Unicode forms and curly apostrophes.
2. **Emoji conversion** — converts emoji names to sentiment terms before TF-IDF.
3. **Repeated-letter normalisation** — handles `goooood`, `baaaaad`, and stretched negation.
4. **Contraction expansion** — converts `don't`, `can't`, etc. to full forms.
5. **Longest-first phrase mapping** — handles idioms and multiword Malay/Manglish phrases.
6. **Tokenisation and filler removal** — preserves useful punctuation boundaries and removes
   fillers such as `lah` and `lor`.
7. **Non-standard spelling mapping** — maps `gud`, `guud`, `guudd`, and `gewd` to `good`.
8. **Restricted fuzzy typo correction** — only corrects longer words from a safe sentiment
   whitelist; ordinary words such as `saya`, `dish`, and `damn` are not fuzzily replaced.
9. **Second semantic phrase pass** — after spelling correction, resolves `not gud` →
   `not good` → `bad`.
10. **Contrast focus** — gives extra model weight to the final clause after
    `but/tapi/however/yet`.
11. **Negation scope** — marks remaining negated tokens for NB/SVM and stops at punctuation or
    contrast boundaries.
12. **TF-IDF features** — word 1–3 grams plus character 2–5 grams.
13. **Classification** — Complement Naive Bayes or Linear SVM, selected by the UI.
14. **Emoji output** — positive 😊, neutral 😐, or negative 😞.

## Labelling and evaluation

- Real-review targets come from high-confidence VADER and TextBlob agreement on readable text.
- Star ratings are not used as sentiment targets.
- Curated English/Malay/Manglish/emoji patterns are training-only.
- The main test set contains unseen real app reviews.
- Latest held-out accuracy: **Naive Bayes 81.19%**, **Linear SVM 96.53%**.

## Important distinction

Readable label text and model feature text are deliberately separate. VADER/TextBlob never
receive machine-only `NOT_` tokens. The NB/SVM models receive explicit negation and contrast
features after the readable text has been prepared.

## Limitations

- Sarcasm and indirect humour remain difficult.
- Unlisted slang and regional dialect vocabulary may remain unknown.
- Fuzzy correction is intentionally conservative to prevent false word replacements.
- Challenge accuracy is diagnostic and must not be presented as universal multilingual accuracy.

