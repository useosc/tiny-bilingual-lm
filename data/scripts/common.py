"""Shared helpers for the data pipeline: normalization, filters, stats."""
from __future__ import annotations

import hashlib
import re
import unicodedata

CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LETTER_RE = re.compile(r"[A-Za-z]")
ALNUM_RE = re.compile(r"[^\W_]", re.UNICODE)  # letters + digits, C-speed
WORD_RE = re.compile(r"[A-Za-z']+")
HTML_TAG_RE = re.compile(r"<[^>]{1,200}>")
CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
MULTI_NL_RE = re.compile(r"\n{3,}")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")

GOOD_EXTRA = " \n\t.,;:!?'\"()[]-—、。，；：！？“”‘’《》…"

EN_STOPWORDS = {
    "the", "and", "of", "to", "in", "is", "that", "for", "it", "as", "with",
    "was", "on", "are", "be", "this", "have", "from", "or", "by", "not",
}
ZH_STOPCHARS = "的了是在和有我你他她它们这那就不人都一上也很到说要去会着没看好自己"

SPAM_PATTERNS = [
    "lorem ipsum", "javascript", "enable javascript", "cookie policy",
    "点击查看", "扫码关注", "微信公众号", "版权所有", "免责声明",
]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CTRL_RE.sub("", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def doc_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _dup_line_ratio(text: str) -> float:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) < 6:
        return 0.0
    counts: dict[str, int] = {}
    for ln in lines:
        counts[ln] = counts.get(ln, 0) + 1
    return max(counts.values()) / len(lines)


def quality_ok(text: str, lang: str) -> tuple[bool, str]:
    """Return (keep, reason_if_dropped). All hot loops use C-level regex/count."""
    n = len(text)
    if n < 200:
        return False, "too_short"
    if n > 30000:
        return False, "too_long"

    cjk = len(CJK_RE.findall(text))
    if lang == "zh":
        if cjk / n < 0.20:
            return False, "not_zh"
        if not any(c in text for c in ZH_STOPCHARS):
            return False, "no_zh_stopchars"
    else:
        if cjk > 0.02 * n:
            return False, "cjk_in_en"
        letters = len(LETTER_RE.findall(text))
        if letters / n < 0.55:
            return False, "not_en"
        sample = text[:5000].lower()
        words = WORD_RE.findall(sample)
        if len(words) < 30:
            return False, "too_few_words"
        hits = sum(1 for w in words if w in EN_STOPWORDS)
        if hits / len(words) < 0.05:
            return False, "no_en_stopwords"

    # Printable ratio, counted with C-level primitives.
    good = len(ALNUM_RE.findall(text))
    good += text.count(" ") + text.count("\n") + text.count("\t")
    for c in GOOD_EXTRA:
        if c in " \n\t":
            continue
        good += text.count(c)
    if good / n < 0.80:
        return False, "symbol_ratio"

    if _dup_line_ratio(text) > 0.30:
        return False, "duplicate_lines"

    low = text[:2000].lower()
    if any(p in low for p in SPAM_PATTERNS):
        return False, "spam_pattern"

    return True, ""
