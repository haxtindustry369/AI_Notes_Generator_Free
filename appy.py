# app.py - Final error-proof AI Notes Generator (ASCII-only PDF, defensive writer)
import io
import os
import re
import random
import collections
import unicodedata
from fpdf import FPDF
import pdfplumber
import streamlit as st

# -------------------------
# Small utils & stopwords
# -------------------------
_STOPWORDS = {
    "the","and","for","that","with","this","from","are","was","were","will","can",
    "but","not","have","has","had","which","what","when","where","how","why","a",
    "an","in","on","of","to","is","as","by","or","be","its","at","into","we","they",
    "their","it","you","your","i","he","she","them","these","those","also","our","may"
}

def _simple_tokenize_words(text):
    return re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

def _sentence_split(text):
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if s.strip()]

def _break_long(text: str, max_len: int = 30) -> str:
    """Insert spaces into very long unbroken sequences to let FPDF wrap them."""
    def _chunk(match):
        s = match.group(0)
        return " ".join(s[i:i+max_len] for i in range(0, len(s), max_len))
    return re.sub(r'\S{' + str(max_len) + r',}', _chunk, text)

def _sanitize_ascii(text: str) -> str:
    """Normalize and strip non-ascii characters; then break long sequences."""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii", "ignore")
    return _break_long(text, max_len=30)

# -------------------------
# Extraction and summarization
# -------------------------
def extract_text_from_pdf(file_stream):
    text_parts = []
    try:
        with pdfplumber.open(file_stream) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
    except Exception:
        try:
            raw = file_stream.read()
            text_parts.append(raw.decode("utf-8", errors="ignore"))
        except Exception:
            pass
    return "\n".join(text_parts)

def extract_text(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf") or uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    else:
        raw = uploaded_file.read()
        if isinstance(raw, bytes):
            try:
                return raw.decode("utf-8", errors="ignore")
            except:
                return raw.decode("latin-1", errors="ignore")
        return str(raw)

def simple_summary(text, n=5):
    sents = _sentence_split(text)
    if not sents:
        return ""
    words = _simple_tokenize_words(text)
    freq = collections.Counter(w for w in words if w not in _STOPWORDS)
    if not freq:
        return "\n".join(sents[:n])
    scores = []
    for s in sents:
        toks = _simple_tokenize_words(s)
        score = sum(freq.get(w, 0) for w in toks)
        scores.append((s, score, len(toks)))
    scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    chosen = [s for s,_,_ in scores[:n]]
    # keep original order
    return "\n".join([s for s in sents if s in chosen])

def extract_frequent_words(text, k=50):
    toks = _simple_tokenize_words(text)
    freq = collections.Counter(t for t in toks if t not in _STOPWORDS)
    return [w for w,_ in freq.most_common(k)]

# -------------------------
# MCQ and Q generation
# -------------------------
def generate_mcqs(text, count=6):
    sents = _sentence_split(text)
    candidates = [s for s in sents if len(s.split()) > 6]
    random.shuffle(candidates)
    freq_words = extract_frequent_words(text, 200)
    mcqs = []
    for s in candidates:
        s_lower = s.lower()
        possible = [w for w in freq_words if re.search(r"\b" + re.escape(w) + r"\b", s_lower)]
        if possible:
            ans = random.choice(possible)
        else:
            opts = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", s) if w.lower() not in _STOPWORDS]
            if not opts:
                continue
            ans = random.choice(opts).lower()
        distractors = [w for w in freq_words if w != ans]
        random.shuffle(distractors)
        distractors = distractors[:3]
        if len(distractors) < 3:
            extras = [w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", s) if w.lower() != ans]
            for e in extras:
                if e not in distractors:
                    distractors.append(e)
                if len(distractors) >= 3:
                    break
        if len(distractors) < 3:
            continue
        q = re.sub(r"\b" + re.escape(ans) + r"\b", "_____", s, flags=re.IGNORECASE)
        opts = [ans] + distractors[:3]
        disp = [o.capitalize() for o in opts]
        random.shuffle(disp)
        correct_index = next(i for i,o in enumerate(disp) if o.lower() == ans.lower())
        mcqs.append({"question": q, "options": disp, "answer": chr(65+correct_index), "answer_text": ans})
        if len(mcqs) >= count:
            break
    return mcqs

def generate_important_questions(text, count=8):
    sents = _sentence_split(text)
    long_sents = [s for s in sents if len(s) > 80]
    long_sents.sort(key=len, reverse=True)
    qs = []
    for s in long_sents[:count]:
        qs.append("Explain: " + (s[:300] + ("..." if len(s) > 300 else "")))
    if not qs:
        fq = extract_frequent_words(text, count)
        qs = [f"Discuss the importance of '{w}' in this topic." for w in fq]
    return qs[:count]

def generate_short_answers(text, count=8):
    sents = _sentence_split(text)
    keys = ['is', 'are', 'means', 'refers to', 'defined as', 'consists of', 'used to']
    chosen = []
    for s in sents:
        if any(k in s.lower() for k in keys) and len(s.split()) < 40:
            chosen.append(s.strip())
        if len(chosen) >= count:
            break
    if len(chosen) < count:
        for s in sents:
            if s.strip() and s.strip() not in chosen:
                chosen.append(s.strip())
            if len(chosen) >= count:
                break
    pairs = []
    for s in chosen[:count]:
        q = f"What is: {s.split('.')[0][:120]}?"
        pairs.append({"q": q, "a": s})
    return pairs

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Notes Generator (Stable)", page_icon="📝", layout="wide")
st.title("AI Notes Generator — Stable (ASCII-only PDF)")

st.sidebar.header("Options")
summary_len = st.sidebar.slider("Summary sentences", 3, 12, 6)
mcq_count = st.sidebar.slider("Number of MCQs", 1, 12, 6)
imp_count = st.sidebar.slider("Important Questions", 1, 12, 6)
sa_count = st.sidebar.slider("Short Answers", 1, 12, 6)
enable_pdf = st.sidebar.checkbox("Enable PDF download", True)

uploaded = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

# Defensive PDF writer helpers (will be used to avoid FPDF width errors)
def _split_to_chunks(s: str, max_chars: int = 150):
    parts = []
    cur = ""
    for tok in s.split(" "):
        if cur == "":
            cur = tok
        elif len(cur) + 1 + len(tok) <= max_chars:
            cur = cur + " " + tok
        else:
            parts.append(cur)
            cur = tok
    if cur:
        parts.append(cur)
    return parts

def _write_defensively(pdf_obj, text_line: str, h=5, max_chunk=120):
    try:
        pdf_obj.multi_cell(0, h, text_line)
        return
    except Exception:
        chunks = _split_to_chunks(text_line, max_chars=max_chunk)
        for chunk in chunks:
            try:
                pdf_obj.multi_cell(0, h, chunk)
            except Exception:
                start = 0
                L = len(chunk)
                while start < L:
                    end = min(start + max(30, max_chunk//4), L)
                    piece = chunk[start:end]
                    try:
                        pdf_obj.multi_cell(0, h, piece)
                    except Exception:
                        for ch in piece:
                            try:
                                pdf_obj.multi_cell(0, h, ch)
                            except Exception:
                                continue
                    start = end

# Main UI flow
if uploaded:
    st.info(f"Uploaded: {uploaded.name}")
    raw_text = extract_text(uploaded)
    if not raw_text or not raw_text.strip():
        st.error("No readable text found. Scanned PDFs need OCR (not included).")
        st.stop()

    with st.expander("Preview (first 2000 chars)"):
        st.write(raw_text[:2000] + ("..." if len(raw_text) > 2000 else ""))

    if st.button("Generate"):
        summary = simple_summary(raw_text, summary_len)
        mcqs = generate_mcqs(raw_text, mcq_count)
        imp_qs = generate_important_questions(raw_text, imp_count)
        short_ans = generate_short_answers(raw_text, sa_count)
        probable = [f"Why is '{w}' important?" for w in extract_frequent_words(raw_text, 8)]

        # Show outputs
        st.header("Summary")
        st.write(summary)

        st.header("MCQs")
        if mcqs:
            for i,m in enumerate(mcqs,1):
                st.markdown(f"**Q{i}.** {m['question']}")
                for oi,opt in enumerate(m['options']):
                    st.write(f"- {chr(65+oi)}. {opt}")
                st.write(f"**Answer:** {m['answer']} — {m['answer_text']}")
        else:
            st.write("No MCQs generated.")

        st.header("Important Questions")
        for i,q in enumerate(imp_qs,1):
            st.write(f"{i}. {q}")

        st.header("Short Answers")
        for i,p in enumerate(short_ans,1):
            st.write(f"{i}. Q: {p['q']}")
            st.write(f"   A: {p['a']}")

        st.header("Most Probable Questions")
        for i,q in enumerate(probable,1):
            st.write(f"{i}. {q}")

        # -------------------------
        # Prepare txt download (original content)
        # -------------------------
        result_txt = f"=== SUMMARY ===\n{summary}\n\n=== MCQS ===\n"
        for idx,m in enumerate(mcqs,1):
            result_txt += f"Q{idx}. {m['question']}\n"
            for oi,opt in enumerate(m['options']):
                result_txt += f"  {chr(65+oi)}. {opt}\n"
            result_txt += f"Answer: {m['answer_text']}\n\n"
        result_txt += "\n=== IMPORTANT QUESTIONS ===\n" + "\n".join(imp_qs) + "\n\n"
        result_txt += "=== SHORT ANSWERS ===\n" + "\n".join([f"Q{i+1}. {p['q']}\nA: {p['a']}" for i,p in enumerate(short_ans)]) + "\n\n"
        result_txt += "=== MOST PROBABLE ===\n" + "\n".join(probable)

        st.download_button("Download TXT", result_txt, file_name=f"{uploaded.name}_notes.txt", mime="text/plain")

        # -------------------------
        # PDF: ASCII only, defensive writing (guaranteed)
        # -------------------------
        if enable_pdf:
            pdf_buf = io.BytesIO()
            pdf = FPDF()
            pdf.add_page()
            # use standard built-in font to prevent any TTF/Unicode issues
            pdf.set_font("Arial", size=10)
            pdf.set_left_margin(10)
            pdf.set_right_margin(10)
            pdf.set_auto_page_break(auto=True, margin=12)

            # prepare ASCII-safe text for PDF
            safe_text_blocks = []
            safe_text_blocks.append("AI Notes (Generated)\n")
            safe_text_blocks.append("SUMMARY:\n" + _sanitize_ascii(summary) + "\n\n")
            safe_text_blocks.append("MCQS:\n")
            for idx,m in enumerate(mcqs,1):
                safe_text_blocks.append(_sanitize_ascii(f"{idx}. {m['question']}"))
                for oi,opt in enumerate(m['options']):
                    safe_text_blocks.append(_sanitize_ascii(f"   {chr(65+oi)}. {opt}"))
                safe_text_blocks.append(_sanitize_ascii(f"   Answer: {m['answer_text']}\n"))
            safe_text_blocks.append("\nImportant Questions:\n" + _sanitize_ascii("\n".join(imp_qs)) + "\n\n")
            safe_text_blocks.append("Short Answers:\n")
            for i,p in enumerate(short_ans,1):
                safe_text_blocks.append(_sanitize_ascii(f"Q{i}. {p['q']}"))
                safe_text_blocks.append(_sanitize_ascii(f"A: {p['a']}\n"))
            safe_text_blocks.append("Most Probable Questions:\n" + _sanitize_ascii("\n".join(probable)))

            # write defensively
            for block in safe_text_blocks:
                # break into lines and write each line defensively
                for line in block.splitlines():
                    if not line.strip():
                        pdf.ln(4)
                        continue
                    # split huge lines into manageable parts then write
                    for part in _split_to_chunks(line, max_chars=150):
                        _write_defensively(pdf, part, h=5, max_chunk=120)

            pdf.output(pdf_buf)
            pdf_buf.seek(0)
            st.download_button("Download PDF", pdf_buf, file_name=f"{uploaded.name}_notes.pdf", mime="application/pdf")
else:
    st.info("Upload a PDF or TXT file to start.")
