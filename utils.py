"""
utils.py
--------
Rule-based helper functions that sit alongside the ML model to produce
the richer analysis features: suspicious claim detection, source
analysis, credibility scoring, and human-readable explanations.

These are intentionally simple, transparent, keyword/regex based
heuristics (not another ML model) so that in a viva you can explain
*exactly* why a claim was flagged — this is what "explainable AI"
means at a minor-project level.
"""

import re
from urllib.parse import urlparse

# ---------------------------------------------------------------------
# 1. Suspicious claim / clickbait detection
# ---------------------------------------------------------------------

SENSATIONAL_WORDS = [
    "shocking", "breaking", "you won't believe", "secret", "exposed",
    "hidden truth", "they don't want you to know", "miracle", "hoax",
    "cover-up", "coverup", "conspiracy", "banned", "deleted", "leaked",
    "whistleblower", "wake up", "goes viral", "forward this", "share before",
    "urgent warning", "stunned", "speechless", "unbelievable", "won't tell you",
]

CLICKBAIT_PHRASES = [
    "click here", "before it's too late", "must see", "won't believe your eyes",
    "number will shock you", "this one trick", "doctors hate", "what happened next",
]

CALL_TO_ACTION_PHRASES = [
    "share this", "forward this", "forward to everyone", "send to 10 friends",
    "share before", "forward now", "share with all", "spread the word",
]

VAGUE_ATTRIBUTION = [
    "sources say", "anonymous source", "someone said", "people are saying",
    "reports suggest", "insiders reveal", "unverified", "unconfirmed reports",
    "according to a whatsapp", "viral message",
]


def _count_hits(text_lower, phrase_list):
    hits = []
    for phrase in phrase_list:
        if phrase in text_lower:
            hits.append(phrase)
    return hits


def detect_suspicious_claims(text: str):
    """
    Scans the article text for known fake-news writing patterns.
    Returns a list of {type, evidence, description} findings plus a
    numeric suspicion score (0-100, higher = more suspicious).
    """
    text_lower = text.lower()
    findings = []

    sensational_hits = _count_hits(text_lower, SENSATIONAL_WORDS)
    if sensational_hits:
        findings.append({
            "type": "Sensational Language",
            "evidence": ", ".join(sorted(set(sensational_hits))[:6]),
            "description": "Uses emotionally charged or sensational wording often "
                            "used to provoke a reaction rather than inform.",
        })

    clickbait_hits = _count_hits(text_lower, CLICKBAIT_PHRASES)
    if clickbait_hits:
        findings.append({
            "type": "Clickbait Phrasing",
            "evidence": ", ".join(sorted(set(clickbait_hits))[:6]),
            "description": "Contains phrasing designed to bait clicks rather than "
                            "convey verified information.",
        })

    cta_hits = _count_hits(text_lower, CALL_TO_ACTION_PHRASES)
    if cta_hits:
        findings.append({
            "type": "Viral Call-to-Action",
            "evidence": ", ".join(sorted(set(cta_hits))[:6]),
            "description": "Explicitly urges the reader to share/forward the message "
                            "quickly — a common trait of unverified chain messages.",
        })

    vague_hits = _count_hits(text_lower, VAGUE_ATTRIBUTION)
    if vague_hits:
        findings.append({
            "type": "Vague / Anonymous Attribution",
            "evidence": ", ".join(sorted(set(vague_hits))[:6]),
            "description": "Claims are attributed to unnamed or unverifiable sources "
                            "instead of a named, checkable authority.",
        })

    exclam_count = text.count("!")
    if exclam_count >= 3:
        findings.append({
            "type": "Excessive Punctuation",
            "evidence": f"{exclam_count} exclamation marks",
            "description": "Heavy use of exclamation marks is a common stylistic "
                            "marker of sensationalized or manipulative writing.",
        })

    caps_words = re.findall(r"\b[A-Z]{4,}\b", text)
    if len(caps_words) >= 2:
        findings.append({
            "type": "Excessive Capitalization",
            "evidence": ", ".join(sorted(set(caps_words))[:6]),
            "description": "Multiple ALL-CAPS words are frequently used to "
                            "artificially emphasize claims.",
        })

    numbers_no_source = re.search(r"\b\d{1,3}%|\bmillions?\b|\bthousands?\b", text_lower)
    named_source = re.search(r"\b(said|reported by|study by|according to)\b", text_lower)
    if numbers_no_source and not named_source:
        findings.append({
            "type": "Unattributed Statistic",
            "evidence": numbers_no_source.group(0),
            "description": "A statistic is mentioned but no named, checkable source "
                            "is attributed to it.",
        })

    # Suspicion score: weighted count of finding types, capped at 100
    score = min(100, len(findings) * 16)

    return findings, score


# ---------------------------------------------------------------------
# 2. Source / URL analysis
# ---------------------------------------------------------------------

KNOWN_CREDIBLE_DOMAINS = {
    "pib.gov.in", "reuters.com", "apnews.com", "bbc.com", "who.int",
    "rbi.org.in", "aajtak.in", "indianexpress.com", "ndtv.com", "livemint.com",
    "timesofindia.indiatimes.com", "isro.gov.in", "prsindia.org", "thehindu.com",
    "hindustantimes.com", "pib.nic.in", "un.org", "nature.com", "sciencedirect.com",
    "gov.in", "nic.in", "edu",
}

KNOWN_LOW_CREDIBILITY_PATTERNS = [
    "blogspot.", "wixsite.", ".xyz", ".click", ".top", ".win", ".info",
    "viral", "shocking", "truth4all", "exposed", "whatsapp-forward",
    "sharewithall", "unbelievable", "newsflash-alert",
]


def analyze_source(source: str):
    """
    Accepts a raw URL or bare domain string (may be empty) and returns
    a structured credibility assessment.
    """
    if not source or not source.strip():
        return {
            "provided": False,
            "domain": None,
            "verdict": "No source provided",
            "detail": "No URL or publication name was given, so source "
                      "credibility could not be assessed. Always check who "
                      "originally published a claim before trusting it.",
            "credibility": "unknown",
        }

    raw = source.strip()
    if "://" not in raw:
        raw_for_parse = "http://" + raw
    else:
        raw_for_parse = raw

    try:
        domain = urlparse(raw_for_parse).netloc.lower()
        domain = domain[4:] if domain.startswith("www.") else domain
    except Exception:
        domain = raw.lower()

    if not domain:
        domain = raw.lower()

    if any(domain == d or domain.endswith("." + d) for d in KNOWN_CREDIBLE_DOMAINS):
        return {
            "provided": True,
            "domain": domain,
            "verdict": "Recognized, generally credible source",
            "detail": f"'{domain}' matches a known established news outlet, "
                      f"government body, or institutional domain.",
            "credibility": "high",
        }

    if any(pattern in domain for pattern in KNOWN_LOW_CREDIBILITY_PATTERNS):
        return {
            "provided": True,
            "domain": domain,
            "verdict": "Matches patterns common in low-credibility sites",
            "detail": f"'{domain}' contains patterns (generic blog hosting, unusual "
                      f"TLD, or sensational naming) often seen in disinformation sites. "
                      f"This is a heuristic signal, not a definitive judgment.",
            "credibility": "low",
        }

    return {
        "provided": True,
        "domain": domain,
        "verdict": "Unverified / not in known source list",
        "detail": f"'{domain}' is not in our list of recognized credible outlets "
                  f"or known low-credibility sites. Manually verify this source "
                  f"through independent search before trusting it.",
        "credibility": "unverified",
    }


# ---------------------------------------------------------------------
# 3. Evidence / fact-check pointers (static, offline-friendly)
# ---------------------------------------------------------------------

FACT_CHECK_SITES = [
    {"name": "PIB Fact Check (Govt. of India)", "url": "https://pib.gov.in/factcheck.aspx"},
    {"name": "Alt News", "url": "https://www.altnews.in/"},
    {"name": "Boom Live", "url": "https://www.boomlive.in/"},
    {"name": "Snopes", "url": "https://www.snopes.com/"},
    {"name": "PolitiFact", "url": "https://www.politifact.com/"},
    {"name": "Google Fact Check Explorer", "url": "https://toolbox.google.com/factcheck/explorer"},
]


def suggest_evidence_sources(topic_keywords):
    """
    Since this project runs fully offline (no live fact-check API key),
    we surface a curated, static list of reputable fact-checking
    platforms and suggest a ready-made search query the user can run
    on them manually. This keeps the 'Evidence Collection' feature
    honest about what is and isn't automated.
    """
    query = " ".join(topic_keywords[:6])
    results = []
    for site in FACT_CHECK_SITES:
        results.append({
            "name": site["name"],
            "url": site["url"],
            "suggested_query": query,
        })
    return results


# ---------------------------------------------------------------------
# 4. Explainability: top contributing words from the model
# ---------------------------------------------------------------------

def explain_prediction(text, vectorizer, model, top_n=8):
    """
    Uses the logistic regression coefficients together with the TF-IDF
    weights actually present in this article to show which words
    pushed the prediction toward FAKE vs REAL.
    """
    X = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    coefs = model.coef_[0]  # positive -> class 1 (FAKE), negative -> class 0 (REAL)

    nonzero_idx = X.nonzero()[1]
    contributions = []
    for idx in nonzero_idx:
        tfidf_value = X[0, idx]
        weight = coefs[idx]
        contribution = tfidf_value * weight
        contributions.append((feature_names[idx], contribution))

    contributions.sort(key=lambda x: x[1], reverse=True)
    top_fake_words = [w for w, c in contributions if c > 0][:top_n]

    contributions.sort(key=lambda x: x[1])
    top_real_words = [w for w, c in contributions if c < 0][:top_n]

    return {
        "fake_indicators": top_fake_words,
        "real_indicators": top_real_words,
    }


def extract_keywords(text, top_n=6):
    """Very small stopword-based keyword extractor for evidence search queries."""
    stopwords = set("""
        a an the is are was were be been being to of in on for with and or but
        this that these those it its as at by from has have had will would
        could should can may might not no so than then their they them he she
        his her you your i we our us do does did about into over under again
    """.split())
    words = re.findall(r"[A-Za-z]{3,}", text.lower())
    freq = {}
    for w in words:
        if w in stopwords:
            continue
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[:top_n]]
