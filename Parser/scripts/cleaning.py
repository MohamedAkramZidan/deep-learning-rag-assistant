"""
Full cleaning pipeline for the "Hands-On Machine Learning" JSON (RAG knowledge base).

Usage:
    python clean_document.py cleaned_document.json cleaned_document_v2.json
    python clean_document.py in.json out.json --drop-code        # also drop flattened code
    python clean_document.py in.json out.json --report report.json

Everything in the "noise" table is DROPPED:
  boilerplate, running footers, glued page numbers, leading footnotes, inline footnote
  markers, "???" refs, figure/table/equation captions, dangling URL gaps, garbled math,
  U+FFFD, split identifiers (rejoined), "- " hyphen breaks (rejoined), printed outputs,
  chapter-end exercises, front/back matter.

Chapter number/title are kept as page metadata ("chapter", "chapter_title"), not in the text.
"""
import argparse
import json
import re
import unicodedata
from collections import Counter

MIN_PAGE = 27                # drop everything before this PDF page
MIN_PAGE_CHARS = 100          # drop pages shorter than this after cleaning
DEFAULT_PAGE_OFFSET = 26      # pdf page_number - printed page number (auto-detected if possible)
FOOTNOTE_MAX_SENTENCES = 2    # fallback length of a footnote when no better boundary exists
FOOTNOTE_MAX_CHARS = 700      # a footnote boundary signal must appear within this window
FOOTER_MIN_PAGES = 2          # a footer phrase must end at least this many pages
EXERCISE_SCAN_LAST_PAGES = 5  # exercises are searched in the last N pages of a chapter

BOILERPLATE_RE = re.compile(
    r"With Early Release ebooks,? you get books in their earliest form\s*[-\u2013\u2014]\s*"
    r"the author['\u2019]s raw and unedited content as he or she writes\s*[-\u2013\u2014]\s*so you\s*")

SMALL = {"of", "and", "the", "to", "with", "for", "in", "on", "a", "an", "using",
         "versus", "vs", "or", "from", "by", "at", "as", "via", "into"}

STARTERS = {
    "The", "This", "These", "Those", "That", "Both", "In", "For", "It", "We", "You", "If", "As",
    "When", "Then", "Now", "Note", "Here", "There", "Each", "Some", "Other", "Many", "Most", "Since",
    "So", "But", "However", "Moreover", "Finally", "First", "Second", "Next", "Let's", "Once",
    "After", "Before", "While", "With", "Because", "Instead", "Unfortunately", "Fortunately",
    "Another", "One", "Two", "Using", "Just", "Of", "Also", "Not", "Okay", "Great", "Hopefully",
    "Obviously", "Indeed", "Thus", "Therefore", "Suppose", "Say", "Consider", "Remember", "Imagine",
    "Try", "See", "Check", "To", "At", "On", "From", "By", "A", "An", "Why", "What", "How", "Which",
    "Who", "Where", "Looks", "Sometimes", "Often", "Typically", "Usually", "Lastly", "Well",
}

PROSE_SHORT = {
    "a", "i", "is", "in", "the", "of", "to", "as", "at", "an", "or", "on", "it", "by", "we", "if", "be",
    "do", "so", "no", "up", "my", "me", "he", "us", "and", "for", "are", "can", "not", "you", "has",
    "was", "but", "all", "one", "two", "its", "any", "out", "use", "how", "may", "new", "get", "let",
    "set", "see", "too", "now", "way", "per", "via", "had", "did", "our", "his", "her", "who", "why",
    "yet", "own", "off", "add", "run", "end", "due", "low", "big", "top", "few", "yes", "etc", "e.g",
    "i.e", "then", "than", "this", "that", "with", "from", "into", "only", "also", "each", "will",
}
MATH_WORDS = {"exp", "log", "max", "min", "sign", "argmin", "argmax", "sum", "sgn", "tanh", "ln",
              "step", "heaviside", "ReLU", "softmax"}
MATH_CHARS = re.compile(
    "[\u0370-\u03ff\u2200-\u22ff\u2190-\u21ff\u2070-\u209f\u00d7\u00b7\u2211\u2225\u22ef\u2206"
    "\u2202\u221e\u2113\u2212\u210b\ufffd\u00af\u02c6\u0302]")

ABBREV = {"vs", "e.g", "i.e", "etc", "al", "cf", "approx", "fig", "no", "dr", "mr", "inc", "ex"}

FN_START = re.compile(r"(\d{1,2}) (?=[A-Z\"\u201c(\[])")
LABEL_RE = re.compile(r"(?:Figure|Table|Equation|Example) \d{1,2}-\d{1,2}\. ")
CAPTION_RE = re.compile(r"(Figure|Table|Equation) \d{1,2}-\d{1,2}\. ")

def norm_unicode(t):
    t = unicodedata.normalize("NFKC", t)
    t = t.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u00a0", " ").replace("\u2013", "-").replace("\u2014", " - ")
    t = re.sub(r"[\u200b-\u200f\u2060\ufeff\ue000-\uf8ff]", "", t)
    t = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def is_title_word(tok):
    core = tok.strip("()[]\"'?!,:;")
    if not core:
        return False
    return core[0].isupper() or core in SMALL


def tail_chain(text, max_words=10):
    toks = text.split()
    if not toks or re.search(r"[.!:;,)\]]$", toks[-1]):
        return [], 0
    chain, i = [], len(toks)
    while i > 0 and len(chain) < max_words:
        t = toks[i - 1]
        if not is_title_word(t):
            break
        if chain and re.search(r"[.!?:;,)\]]$", t):
            break
        chain.append(t)
        i -= 1
    chain.reverse()
    while chain and chain[0] in SMALL:          # trim leading small words
        chain.pop(0)
    return chain, len(toks) - len(chain)


def prev_is_boundary(toks, idx):
    return idx == 0 or bool(re.search(r"[.!?:;)\]\"]$", toks[idx - 1]))


def _abbrev_before(text, end):
    m = re.search(r"([A-Za-z.]+)\W*$", text[:end])
    return bool(m) and m.group(1).strip(".").lower() in ABBREV


def footnote_end(text, start, nxt):
    win = text[start:start + FOOTNOTE_MAX_CHARS * 4]
    cands = []
    m = re.search(r"(?<=[.!?)\]\"])\s+(%d) (?=[A-Z\"(\[])" % nxt, win[:FOOTNOTE_MAX_CHARS])
    if m:
        cands.append((start + m.start(1), "next"))
    m = LABEL_RE.search(win[:FOOTNOTE_MAX_CHARS])
    if m:
        cands.append((start + m.start(), "label"))
    for m in re.finditer(r"[.!?][\")]?\s+(?=[a-z_])", win[:FOOTNOTE_MAX_CHARS]):
        if _abbrev_before(win, m.start() + 1):
            continue
        cands.append((start + m.end(), "lower"))
        break
    if cands:
        return min(cands)
    ends = [m.end() for m in re.finditer(r"[.!?][\")]?(?=\s|$)", win) if not _abbrev_before(win, m.end())]
    if ends:
        return start + ends[min(FOOTNOTE_MAX_SENTENCES, len(ends)) - 1], "cap"
    return len(text), "cap"


def strip_leading_footnotes(text):
    nums, pos, expected, uncertain = set(), 0, None, False
    while True:
        m = FN_START.match(text, pos)
        if not m:
            break
        n = int(m.group(1))
        if expected is not None and n != expected:
            break
        if expected is None and n > 30:
            break
        nums.add(n)
        expected = n + 1
        end, reason = footnote_end(text, m.end(), expected)
        pos = end
        while pos < len(text) and text[pos] == " ":
            pos += 1
        if reason != "next":
            uncertain = reason == "cap"
            break
    return text[pos:].lstrip(), nums, uncertain


def remove_inline_markers(text, nums):
    if nums:
        alt = "|".join(sorted((str(n) for n in nums), key=len, reverse=True))
        text = re.sub(r"(?<=[A-Za-z]{4})(?:%s)(?=\s|[.,;:)])" % alt, "", text)
        text = re.sub(r"(?<=[A-Za-z)\]][.,;:])(?:%s)(?=\s)" % alt, "", text)
    return re.sub(r"(?<=[a-z)]\.)\d{1,2}(?=\s+[A-Z])", "", text)


def find_footer(text, counts):
    chain, start_idx = tail_chain(text)
    if not chain:
        return None
    toks = text.split()
    for n in range(len(chain), 0, -1):
        suffix = " ".join(chain[-n:])
        if counts.get(suffix, 0) < FOOTER_MIN_PAGES:
            continue
        idx = len(toks) - n
        if n >= 3 or prev_is_boundary(toks, idx):
            return suffix
    return None


def build_footer_counts(texts):
    counts = Counter()
    for t in texts:
        chain, _ = tail_chain(t)
        for n in range(1, len(chain) + 1):
            counts[" ".join(chain[-n:])] += 1
    return counts


def detect_offset(pages):
    c = Counter()
    for p in pages:
        m = re.search(r"[A-Za-z]{2,}(\d{2,3})\s*$", p["text"])
        if m:
            c[p["page_number"] - int(m.group(1))] += 1
    return c.most_common(1)[0][0] if c else DEFAULT_PAGE_OFFSET


def strip_page_number_tail(text, page_number, offset):
    """returns (text, fragment) - fragment is a word piece to glue to the next page."""
    m = re.search(r"([A-Za-z]{2,})(\d{1,3})\s*$", text)
    if m and page_number - int(m.group(2)) == offset:
        return text[:m.start()].rstrip(), m.group(1)
    m = re.search(r"\s(\d{1,3})\s*$", text)
    if m and page_number - int(m.group(1)) == offset:
        return text[:m.start()].rstrip(), ""
    return text, None


def strip_page_number_head(text, page_number, offset):
    m = re.match(r"(\d{1,3})\s+", text)
    if m and page_number - int(m.group(1)) == offset:
        return text[m.end():]
    return text


def find_exercise_start(texts):
    """texts = list of page texts of one chapter. returns (page_idx, char_idx) or None"""
    lo = max(0, len(texts) - EXERCISE_SCAN_LAST_PAGES)
    for i in range(lo, len(texts)):
        t = texts[i]
        for m in re.finditer(r"(?:^|(?<=[.?!:)] ))1\. (?=[A-Z])", t):
            tail = t[m.start():] + " " + " ".join(texts[i + 1:])
            if re.search(r"\s2\. [A-Z]", tail) and re.search(r"\s3\. [A-Z]", tail):
                cut = m.start()
                head = t[:cut].rstrip()
                if head.endswith(":"):                 # drop the lead-in sentence too
                    k = max(head.rfind(". "), head.rfind("? "), head.rfind("! "))
                    head = head[:k + 1] if k != -1 else ""
                return i, head
    return None

def remove_captions(text):
    out, pos = [], 0
    for m in CAPTION_RE.finditer(text):
        if m.start() < pos:
            continue
        before = text[:m.start()].rstrip()
        if before and not re.search(r"[.!?:;)\]\"]$|\d$", before):
            continue                                    # a reference like "see Figure 3-3. ..."
        rest = text[m.end():]
        toks = rest.split(" ")
        limit = min(len(toks), 16)
        b = None
        for j in range(limit):
            t = toks[j]
            if t.endswith("?"):
                b = j + 1
                break
            if j >= 1:
                p = toks[j - 1]
                lowerish = (p[:1].islower() or p.endswith(")") or re.search(r"[a-z]\d{1,2}$", p))
                if lowerish and t[:1].isupper():
                    b = j
                    break
        if b is None:
            for j in range(1, limit):
                if toks[j].strip(",") in STARTERS:
                    b = j
                    break
        if b is None:
            b = len(toks) if len(toks) <= 16 else 0
        out.append(text[pos:m.start()])
        pos = m.end() + len(" ".join(toks[:b])) + (1 if b else 0)
    out.append(text[pos:])
    return "".join(out)


def _is_math_tok(tok):
    core = tok.strip(".,;:()[]\"'")
    if not core:
        return False
    if MATH_CHARS.search(tok):
        return True
    if core in MATH_WORDS:
        return True
    if re.fullmatch(r"[\d.,\-+/*^=<>|]+", core):
        return True
    if len(core) <= 3 and core.lower() not in PROSE_SHORT and re.fullmatch(r"[A-Za-z0-9_]+", core):
        return True
    return False


def remove_math_runs(text, min_run=6, min_symbols=2):
    toks = text.split(" ")
    flags = [_is_math_tok(t) for t in toks]
    keep, i, n = [], 0, len(toks)
    while i < n:
        if not flags[i]:
            keep.append(toks[i])
            i += 1
            continue
        j = i
        while j < n and flags[j]:
            j += 1
        run = toks[i:j]
        symbols = sum(1 for t in run if MATH_CHARS.search(t) or t in {"=", "==", "+", "-"})
        if len(run) >= min_run and symbols >= min_symbols:
            pass                                          # dropped
        else:
            keep.extend(run)
        i = j
    return " ".join(keep)

def strip_calls(text, names=("array", "dict_keys", "dict_values")):
    pat = re.compile(r"(?<![\w.])(?:%s)\(" % "|".join(names))
    out, pos = [], 0
    while True:
        m = pat.search(text, pos)
        if not m:
            out.append(text[pos:])
            break
        out.append(text[pos:m.start()])
        depth, j = 1, m.end()
        while j < len(text) and depth:
            depth += (text[j] == "(") - (text[j] == ")")
            j += 1
        pos = j
    return "".join(out)


OUTPUT_PATTERNS = [
    r"\[\s*(?:<(?:tf|tensorflow)[^<>]*>,?\s*)+\]",
    r"<(?:tf|tensorflow)[^<>]*>",
    r"Train on \d+ samples, validate on \d+ samples",
    r"Epoch \d+/\d+\s+(?:\d+/\d+\s*\[[=>.\s]*\]\s*-\s*)?.*?(?:val_)?(?:acc|accuracy|loss|mse)\w*:\s*"
    r"[\d.eE+\-]+(?=\s+(?:Epoch|[A-Z])|\s*$)",
    r"\d+/\d+\s*\[[=>.]+\]\s*-\s*ETA:\s*\S+\s*-\s*loss:\s*[\d.]+(?:\s*-\s*acc:\s*[\d.]+)?",
    r"\[[\s\d.,eE+\-]{30,}\]",
    r"Scores:\s*(?=Mean:)",
    r"Mean:\s*[\d.]+\s*Standard deviation:\s*[\d.]+",
    r"Name: [\w\-]+, dtype: \w+",
    r"\d+\.\d{5,}\s*\{[^{}]*\}",
    r"-?\b\d+\.\d{7,}\b",
    r"pip \d+\.\d+(?:\.\d+)? from \S+ \(python [\d.]+\)",
    r"Collecting \S+(?: Downloading \S+)?",
    r"Successfully installed \S+",
    r"Using base prefix '[^']*'",
    r"New python executable in \S+",
    r"Also creating executable in \S+",
    r"Installing setuptools, pip, wheel\.\.\.done\.",
    r"\[\.\.\.\]",
]
OUTPUT_RES = [re.compile(p) for p in OUTPUT_PATTERNS]


def remove_outputs(text):
    text = strip_calls(text)
    for rx in OUTPUT_RES:
        text = rx.sub(" ", text)
    return text

def remove_refs_and_gaps(text):
    text = re.sub(r"\s*\((?:see )?\?\?\?[^()]*\)", "", text)
    text = re.sub(r"\s+(?:see|in|at|and|to)\s+\?\?\?(?:\s+for more details)?", "", text)
    text = text.replace("???", "")
    text = re.sub(r"\b(?:in|at|see|out)\s+(?=[;,.)])", "", text)
    # URL that was stripped earlier leaves "at Next ..." / "to Next ..." etc.
    text = re.sub(r"(?<=\w) (?:at|to|from|visit) (?=(?:Next|Then|Now|First|Finally|Once|You|It|If|We|Create|Go)\b)",
                  ". ", text)
    text = re.sub(r"(?:\s-){2,}(?=\s|$)", " ", text)       # "- - -" from stripped bullet URLs
    return text


def build_vocab(texts):
    words, idents = Counter(), set()
    for t in texts:
        for w in re.findall(r"[A-Za-z][A-Za-z0-9_\-]*", t):
            words[w.lower()] += 1
        for tok in t.split(" "):
            tok = tok.lstrip("\"'([{<")
            m = re.match(r"[A-Za-z_][A-Za-z0-9_.]*", tok)
            if not m:
                continue
            h = m.group(0).rstrip(".")
            if len(h) >= 6 and (("_" in h) or re.search(r"[A-Za-z]\.[A-Za-z]", h)
                                or re.search(r"[a-z][A-Z]", h)):
                idents.add(h)
    return words, idents


def rejoin_identifiers(text, idents):
    toks, out, i = text.split(" "), [], 0
    while i < len(toks):
        a = toks[i]
        if i + 1 < len(toks):
            b = toks[i + 1]
            ma = re.search(r"[A-Za-z0-9_.]+$", a)
            mb = re.match(r"[A-Za-z0-9_.]+", b)
            if ma and mb:
                cand = ma.group(0) + mb.group(0).rstrip(".")
                if cand in idents and ma.group(0) not in idents and mb.group(0) not in idents:
                    out.append(a[:ma.start()] + ma.group(0) + b)
                    i += 2
                    continue
        out.append(a)
        i += 1
    return " ".join(out)


def rejoin_hyphens(text, words):
    def fix(m, soft):
        a, b = m.group(1), m.group(2)
        if (a + b).lower() in words:
            return a + b
        if f"{a}-{b}".lower() in words:
            return f"{a}-{b}"
        return a + b if soft else f"{a}-{b}"
    text = re.sub(r"(?<![\w-])([A-Za-z][\w]*)- ([A-Za-z][\w]*)", lambda m: fix(m, False), text)
    return re.sub(r"([A-Za-z][\w]*)\u2010 ([A-Za-z][\w]*)", lambda m: fix(m, True), text)


CODE_TOK = re.compile(
    r"^[#@$]|[A-Za-z0-9_\])]\.[A-Za-z_]|\w\(|\)\.|[=\[\]{}<>]|^\*\*|_\w|\w_|^(?:import|def|lambda|elif|None)$")


def drop_code_runs(text, window=10, ratio=0.5):
    toks = text.split(" ")
    flag = [bool(CODE_TOK.search(t)) for t in toks]
    mark = [False] * len(toks)
    for i in range(0, max(1, len(toks) - window + 1)):
        if sum(flag[i:i + window]) >= ratio * window:
            for k in range(i, min(len(toks), i + window)):
                mark[k] = True
    return " ".join(t for t, m in zip(toks, mark) if not m)


def final_tidy(text):
    text = re.sub(r"\(\s*\)|\[\s*\]", " ", text)
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"([,;:])\s*(?=[.])", r"\1", text)
    text = re.sub(r"(?:\. ){2,}", ". ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"^[\s.,;:\-]+", "", text)

def clean(doc, drop_code=False, min_page=MIN_PAGE):
    rep = {"rules": Counter(), "footers": Counter(), "footnote_uncertain_pages": [],
           "dropped_pages": [], "page_offset": None}
    pages = sorted(doc["pages"], key=lambda p: p["page_number"])
    pages = [p for p in pages if p["page_number"] >= min_page]

    # ---- pass A: unicode, chapter openers, footnotes
    recs, ch_no, ch_title = [], 0, None
    for p in pages:
        t = norm_unicode(p["text"])
        t = t.replace("\ufffd", "")
        m = BOILERPLATE_RE.search(t)
        nums, opener = set(), False
        if m:
            rep["rules"]["boilerplate"] += 1
            opener, ch_no = True, ch_no + 1
            pre, body = t[:m.start()].strip(), t[m.end():].strip()
            chain, idx = tail_chain(pre)
            ch_title = " ".join(chain) or ch_title
            foot = pre[:len(pre) - len(" ".join(chain))] if chain else pre
            nums = {int(x) for x in re.findall(r"(?:^|(?<=[.!?)] ))(\d{1,2}) (?=[A-Z\"(])", foot)}
            if foot.strip():
                rep["rules"]["footnote_blocks"] += 1
            t = body
        else:
            t, nums, unc = strip_leading_footnotes(t)
            if nums:
                rep["rules"]["footnote_blocks"] += 1
            if unc:
                rep["footnote_uncertain_pages"].append(p["page_number"])
        before = t
        t = remove_inline_markers(t, nums)
        if t != before:
            rep["rules"]["inline_footnote_markers"] += 1
        recs.append({"page_number": p["page_number"], "text": t, "chapter": ch_no,
                     "chapter_title": ch_title, "opener": opener})

    # ---- pass B: running footers + glued page numbers (state is per page, glue joined later)
    offset = detect_offset(recs)
    rep["page_offset"] = offset
    counts = build_footer_counts([r["text"] for r in recs])
    for r in recs:
        t = r["text"]
        frag = None
        for _ in range(3):
            f = find_footer(t, counts)
            if f:
                t = t[: len(t) - len(f)].rstrip()
                rep["footers"][f] += 1
            t2, fr = strip_page_number_tail(t, r["page_number"], offset)
            if fr is not None:
                frag = fr
                rep["rules"]["glued_page_numbers"] += 1
                t = t2
            if not f and fr is None:
                break
        t = strip_page_number_head(t, r["page_number"], offset)
        r["text"], r["frag"] = t, frag

    # glue fragments across pages: "non" + "zero elements" -> "nonzero elements"
    by_pn = {r["page_number"]: r for r in recs}
    for r in recs:
        if r.get("frag"):
            nxt = by_pn.get(r["page_number"] + 1)
            if nxt:
                nxt["text"] = r["frag"] + nxt["text"]
                rep["rules"]["rejoined_across_pages"] += 1

    # ---- chapter-end exercises
    chapters = {}
    for r in recs:
        chapters.setdefault(r["chapter"], []).append(r)
    for ch, rs in chapters.items():
        hit = find_exercise_start([r["text"] for r in rs])
        if hit:
            i, head = hit
            rs[i]["text"] = head
            for r in rs[i + 1:]:
                r["text"] = ""
            rep["rules"]["exercise_blocks"] += 1

    # ---- pass C: content cleaning
    words, idents = build_vocab([r["text"] for r in recs])
    cleaned = []
    for r in recs:
        t = r["text"]
        if re.match(r"(About the Author|Colophon|Index)\b", t):
            rep["dropped_pages"].append((r["page_number"], "back matter"))
            continue
        t = remove_outputs(t)
        t = remove_math_runs(t)
        t = remove_captions(t)
        t = remove_refs_and_gaps(t)
        t = rejoin_hyphens(t, words)
        t = rejoin_identifiers(t, idents)
        if drop_code:
            t = drop_code_runs(t)
        t = final_tidy(t)
        if len(t) < MIN_PAGE_CHARS:
            rep["dropped_pages"].append((r["page_number"], f"too short ({len(t)} chars)"))
            continue
        cleaned.append({"page_number": r["page_number"], "chapter": r["chapter"],
                        "chapter_title": r["chapter_title"], "text": t})
    return cleaned, rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--drop-code", action="store_true", help="also drop flattened code runs")
    ap.add_argument("--report", default=None)
    ap.add_argument("--min-page", type=int, default=MIN_PAGE)
    a = ap.parse_args()

    doc = json.load(open(a.input, encoding="utf-8"))
    cleaned, rep = clean(doc, drop_code=a.drop_code, min_page=a.min_page)
    out = {"document_id": doc.get("document_id"), "filename": doc.get("filename"),
           "metadata": {**doc.get("metadata", {}), "cleaned_pages": len(cleaned)}, "pages": cleaned}
    json.dump(out, open(a.output, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    rep["rules"] = dict(rep["rules"])
    rep["footers"] = dict(rep["footers"].most_common(60))
    print(f"pages in: {len(doc['pages'])}  pages out: {len(cleaned)}  offset: {rep['page_offset']}")
    print("rules fired:", rep["rules"])
    print("dropped pages:", rep["dropped_pages"][:15], "..." if len(rep["dropped_pages"]) > 15 else "")
    print("footnote boundary uncertain on pages:", rep["footnote_uncertain_pages"][:20])
    if a.report:
        json.dump(rep, open(a.report, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()