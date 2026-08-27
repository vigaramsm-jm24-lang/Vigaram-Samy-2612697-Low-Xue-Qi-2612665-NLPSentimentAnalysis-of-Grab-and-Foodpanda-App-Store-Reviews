"""Shared text normalisation used by training, notebooks, tests, and Streamlit.

The project keeps label preparation separate from model feature preparation.
VADER/TextBlob receive readable text, while the machine-learning models also
receive explicit negation and contrast features.
"""

from __future__ import annotations

import difflib
from functools import lru_cache
import re
import unicodedata

import emoji


CONTRACTIONS = {
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "can't": "cannot", "won't": "will not", "wouldn't": "would not",
    "shouldn't": "should not", "couldn't": "could not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "i'm": "i am", "it's": "it is", "that's": "that is",
}

# Longest phrases are replaced first. The old split()+one-word lookup could
# never match entries such as "tak puas hati" or "tak boleh guna".
PHRASE_MAP = {
    "langsung tak sedap": "not delicious at all",
    "langsung tidak sedap": "not delicious at all",
    "memang tak puas hati": "very dissatisfied",
    "sangat tak puas hati": "very dissatisfied",
    "tak puas hati": "not satisfied", "tidak puas hati": "not satisfied",
    "puas hati": "satisfied", "sakit hati": "very angry",
    "tak boleh guna": "cannot use", "tidak boleh guna": "cannot use",
    "tak boleh": "cannot", "tidak boleh": "cannot",
    "tak bagus": "not good", "tidak bagus": "not good",
    "tak sedap": "not delicious", "tidak sedap": "not delicious",
    "tak sampai": "not arrived", "belum sampai": "not arrived yet",
    "sedap gila": "very delicious", "sedap giler": "very delicious",
    "power gila": "very excellent", "teruk gila": "very terrible",
    "lambat gila": "very late", "takde masalah": "no problem",
    "tiada masalah": "no problem", "takde": "unavailable", "xde": "unavailable",
    "x boleh": "cannot", "xboleh": "cannot",
    "x jadi": "did not happen", "xjadi": "did not happen",
    "no cap": "genuinely good", "not too bad": "acceptable good",
    "damn nice": "very nice", "damn good": "very good",
    "damn delicious": "very delicious",
    "not bad": "acceptable good", "not the best": "somewhat disappointing",
    "could be worse": "acceptable", "no problem": "acceptable good",
    "no issue": "acceptable good", "do not like": "dislike",
    "does not like": "dislike", "did not like": "disliked",
    "do not enjoy": "dislike", "does not enjoy": "dislike",
    "did not enjoy": "disliked", "has not arrived": "missing delivery",
    "have not arrived": "missing delivery",
    "not satisfied": "dissatisfied", "not happy": "unhappy",
    "not delicious": "bad tasting", "not tasty": "bad tasting",
    "not fresh": "stale", "not good": "bad", "not working": "broken",
    "does not work": "broken", "did not work": "broken",
    "cannot use": "unusable", "never arrived": "missing delivery",
    # Implied negativity -- none of these words are individually negative,
    # but the phrase as a whole clearly signals dissatisfaction
    "won't come back": "will not return terrible",
    "wont come back": "will not return terrible",
    "never coming back": "will not return terrible",
    "not coming back": "will not return terrible",
    "tak akan datang lagi": "will not return terrible",
    "takkan datang lagi": "will not return terrible",
    "tak akan order lagi": "will not order terrible",
    "tak nak beli lagi": "will not buy terrible",
    "replaced it quickly": "resolved successfully",
    "fixed the problem quickly": "resolved successfully",
    # Common emoji.demojize output (underscores are changed to spaces first).
    "face vomiting": "disgusting", "nauseated face": "disgusting",
    "enraged face": "angry", "pouting face": "angry", "angry face": "angry",
    "face with symbols on mouth": "angry",
    "smiling face with heart eyes": "love", "face with heart eyes": "love",
    "red heart": "love", "thumbs up": "good", "thumbs down": "bad",
    "crying face": "sad", "loudly crying face": "very sad",
}

WORD_MAP = {
    "sedap": "delicious", "sedapp": "delicious", "bagus": "good",
    "mantap": "excellent", "cun": "nice", "best": "excellent",
    "power": "excellent", "steady": "reliable", "cepat": "fast",
    "pantas": "fast", "murah": "cheap", "comel": "cute",
    "cantik": "beautiful", "teruk": "terrible", "rosak": "broken",
    "geram": "frustrated", "menyampah": "disgusted", "bodoh": "stupid",
    "lembab": "slow", "lambat": "late", "mahal": "expensive",
    "hampeh": "disappointing", "sejuk": "cold", "panas": "hot",
    "basi": "stale", "tak": "not", "tidak": "not", "bukan": "not",
    "tapi": "but", "tetapi": "but", "namun": "but",
    "walaupun": "although", "goat": "amazing", "slay": "excellent",
    "cringe": "embarrassing", "cap": "lie",
    # common informal English spelling variants (documented common usage,
    # not a general spelling-correction system -- vowel substitutions like
    # this can't be safely caught by fuzzy matching without also wrongly
    # matching unrelated words)
    "gud": "good", "guud": "good", "guudd": "good", "gewd": "good",
    "delish": "delicious",
    # Sarawak Malay dialect - "sik" is a genuine, distinct negation word
    # (not a variant spelling of "tak"), confirmed via Sarawak Malay
    # reference sources. Regional dialect vocabulary is limited here to
    # a couple of well-documented words, not comprehensive coverage.
    "sik": "not",
    # "sepad" (common typo of "sedap") added explicitly -- its similarity
    # to "sedap" (0.6) is now below the 0.7 fuzzy-match cutoff we raised
    # to prevent false positives, so it needs an exact entry to still work.
    "sepad": "delicious",
    # "gila"/"giler" is used as an INTENSIFIER ("very"), not literally
    # "crazy", when it follows a sentiment word e.g. "sedap gila" = "very
    # delicious". Restored here -- this was present earlier in the
    # project's history but missing from this rebuild.
    "gila": "very", "giler": "very",
}

FILLERS = {"lah", "lor", "leh", "meh", "wor", "wei", "hor", "eh", "kot", "pun", "lo"}
CONTRAST_MARKERS = ("but", "however", "yet", "nevertheless")
NEGATION_TRIGGERS = {"not", "no", "never", "neither", "cannot"}
CLAUSE_BREAKERS = set(CONTRAST_MARKERS) | {"although", "though", "except", "still"}

# Fuzzy matching is deliberately restricted to longer, typo-safe sentiment
# words. Matching every unknown word against WORD_MAP caused severe false
# positives: the ordinary Malay pronoun "saya" matched internet slang "slay"
# and became "excellent". Short/non-standard forms such as gud/guudd are
# handled by exact WORD_MAP entries instead.
FUZZY_SAFE_MAP = {
    "delicious": "delicious", "excellent": "excellent",
    "terrible": "terrible", "disgusting": "disgusting",
    "frustrated": "frustrated", "disappointing": "disappointing",
    "reliable": "reliable", "beautiful": "beautiful",
    "expensive": "expensive", "unavailable": "unavailable",
    "satisfied": "satisfied", "dissatisfied": "dissatisfied",
}


def _replace_phrases(text: str) -> str:
    for phrase in sorted(PHRASE_MAP, key=len, reverse=True):
        pattern = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        text = re.sub(pattern, PHRASE_MAP[phrase], text, flags=re.IGNORECASE)
    return text


@lru_cache(maxsize=100_000)
def _fuzzy_word_lookup(word: str, cutoff: float = 0.82, min_length: int = 5) -> str:
    """Correct only close typos of a small, safe sentiment-word whitelist."""
    if word in WORD_MAP:
        return WORD_MAP[word]
    if len(word) < min_length:
        return word
    candidates = [
        candidate for candidate in FUZZY_SAFE_MAP
        if candidate[0] == word[0] and abs(len(candidate) - len(word)) <= 2
    ]
    matches = difflib.get_close_matches(word, candidates, n=1, cutoff=cutoff)
    return FUZZY_SAFE_MAP[matches[0]] if matches else word


DOUBLE_LETTER_WORDS = {
    # Common English words with a genuine double letter that someone might
    # stretch for emphasis. A word NOT in this list (or WORD_MAP) defaults
    # to collapsing fully to 1 repeated letter, since most words ("bad",
    # "best", "teruk", "sedap"...) only have a single letter there --
    # collapsing those to 2 would produce a DIFFERENT wrong spelling, not
    # a fix (e.g. "baaaaad" collapsed to 2 gives "baad", not "bad").
    "good", "sweet", "class", "food", "book", "look", "took", "cool",
    "feel", "need", "deep", "still", "less", "well", "tell", "full",
    "call", "fall", "small", "will", "bill", "miss", "pass", "boss",
    "hell", "poor", "soon", "keep", "seem", "free", "green", "happy",
}


@lru_cache(maxsize=100_000)
def _destretch_word(word: str) -> str:
    """Only touches words with 3+ repeated characters (genuine stretching
    like "goooood" or "baaaaad") -- a normal word like "good" only has 2
    o's and is left completely untouched. For an affected word, try
    collapsing to 2 first, but only KEEP that version if it's a real known
    word (WORD_MAP or DOUBLE_LETTER_WORDS); otherwise collapse fully to 1,
    since that's correct for the vast majority of words."""
    collapsed_to_2 = re.sub(r"(.)\1{2,}", r"\1\1", word)
    if collapsed_to_2 in WORD_MAP or collapsed_to_2 in DOUBLE_LETTER_WORDS:
        return collapsed_to_2
    return re.sub(r"(.)\1{2,}", r"\1", word)


def _destretch(text: str) -> str:
    """Apply _destretch_word() to every letter-sequence in the string,
    leaving punctuation/spacing/emoji untouched around it. Without this,
    "goooood" is a completely unknown word to the model -- it never saw
    that exact spelling during training, so it carries zero signal even
    though a human instantly reads it as emphasised "good"."""
    return re.sub(r"[a-zA-Z]+", lambda m: _destretch_word(m.group(0)), text)


def normalize_for_labeler(text: object) -> str:
    """Return readable normalised text for VADER/TextBlob or display."""
    value = unicodedata.normalize("NFKC", str(text))
    value = value.replace("’", "'").replace("‘", "'").replace("`", "'")
    value = emoji.demojize(value, delimiters=(" ", " "))
    value = value.replace("_", " ").replace(":", " ").lower()
    # Destretch AFTER lowercasing -- WORD_MAP/DOUBLE_LETTER_WORDS are
    # lowercase, so "GOOOOOOOOOD" must become "goooooood" BEFORE
    # destretching, or the collapsed "GOOD" would never match "good" and
    # fall through to the wrong "GOD".
    value = _destretch(value)
    for short, full in sorted(CONTRACTIONS.items(), key=lambda item: len(item[0]), reverse=True):
        value = re.sub(r"(?<!\w)" + re.escape(short) + r"(?!\w)", full, value)
    value = re.sub(r"n['’]?t\b", " not", value)
    value = _replace_phrases(value)
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[.,!?;]", value)
    translated = [_fuzzy_word_lookup(token) for token in tokens if token not in FILLERS]
    value = " ".join(translated)
    # Run phrase normalisation a SECOND time after spelling correction.
    # Example: "not gud" cannot match the earlier "not good" phrase until
    # _fuzzy_word_lookup() first turns "gud" into "good".  The second pass
    # now resolves it to the explicit negative token "bad", preventing the
    # strongly positive word "good" from overpowering the negation in NB.
    value = _replace_phrases(value)
    value = re.sub(r"\s+([.,!?;])", r"\1", value)
    return re.sub(r"\s+", " ", value).strip()


def contrast_tail(text: str) -> str | None:
    """Return the clause after the final contrast marker, if present."""
    pattern = r"\b(?:" + "|".join(map(re.escape, CONTRAST_MARKERS)) + r")\b[,:;]?"
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    if len(parts) < 2:
        return None
    tail = parts[-1].strip(" ,;:-")
    return tail if len(re.findall(r"[a-z]+", tail)) >= 2 else None


def focus_for_labeler(text: object) -> str:
    """Give the post-contrast clause priority for weak labelling."""
    normalised = normalize_for_labeler(text)
    return contrast_tail(normalised) or normalised


def _is_negation_trigger(token: str) -> bool:
    """"noooooo" destretches to "noo" (collapse stops at 2, to protect real
    double letters like "good"), which does NOT exactly match "no" -- so
    elongated negation would silently vanish. Short critical words rarely
    have genuine double letters in their base form, so also check the
    fully-collapsed-to-1 version against the trigger list."""
    if token in NEGATION_TRIGGERS:
        return True
    return re.sub(r"(.)\1+", r"\1", token) in NEGATION_TRIGGERS


def _scope_negation(text: str) -> list[str]:
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+|[.,!?;]", text.lower())
    output: list[str] = []
    remaining = 0
    for token in tokens:
        punctuation = token in ".,!?;"
        if _is_negation_trigger(token):
            output.append(token)
            remaining = 4
            continue
        if punctuation or token in CLAUSE_BREAKERS:
            if not punctuation:
                output.append(token)
            remaining = 0
            continue
        if remaining > 0:
            output.append(f"NOT_{token}")
            remaining -= 1
        else:
            output.append(token)
    return output


def preprocess_for_model(text: object) -> str:
    """Create NB/SVM features, including negation and post-contrast focus."""
    normalised = normalize_for_labeler(text)
    result = _scope_negation(normalised)
    tail = contrast_tail(normalised)
    if tail:
        focused = _scope_negation(tail)
        result.extend(["CONTRAST_FOCUS", *focused, "CONTRAST_FOCUS", *focused])
    return " ".join(result).strip()


def sentiment_emoji(label: str) -> str:
    return {"positive": "😊", "neutral": "😐", "negative": "😞"}.get(label, "❓")
