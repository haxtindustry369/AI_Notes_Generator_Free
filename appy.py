# app.py — Final stable version (ASCII-only PDF, no Unicode font)
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
# Helpers: tokenization / stopwords
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

# break very long words so PDF never crashes
def _break_long(text: str, max_len: int = 30) -> str:
    def _chunk(m):
        s = m.group(0)
        return " ".join(s[i:i+max_len] for i in range(0, len(s), max_len))
    return re.sub(r'\S{' + str(max_len) + r',}', _chunk, text)

# sanitize for ASCII PDF
def _sanitize_ascii(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii", "ignore")
    return _break_long(text, max_len=30)

# -------------------------
# Extraction / summarization
# -------------------------
def extract_text_from_pdf(file_stream):
    txt = []
    try:
        with pdfplumber.open(file_stream) as pdf:
            for page in pdf.pages:
                txt.append(page.extract_text() or "")
    except:
        try:
            raw = file_stream.read()
            txt.append(raw.decode("utf-8", errors="ignore"))
        except:
            pass
    return "\n".join(txt)

def extract_text(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
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
        t = _simple_tokenize_words(s)
        score = sum(freq.get(w, 0) for w in t)
        scores.append((s, score, len(t)))
    scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    chosen = [s for s,_,_ in scores[:n]]
    return "\n".join([s for s in sents if s in chosen])

def extract_frequent_words(text, k=50):
    tokens = _simple_tokenize_words(text)
    freq = collections.Counter(t for t in tokens if t not in _STOPWORDS)
    return [w for w,_ in freq.most_common(k)]

# -------------------------
# MCQ / questions
# -------------------------
def generate_mcqs(text, count=6):
    sents = _sentence_split(text)
    candidates = [s for s in sents if len(s.split()) > 6]
    random.shuffle(candidates)
    words = extract_frequent_words(text, 200)
    mcqs = []
    for s in candidates:
        s_low = s.lower()
        possible = [w for w in words if re.search(rf"\b{re.escape(w)}\b", s_low)]
        if possible:
            ans = random.choice(possible)
        else:
            toks = [w for w in re.findall(r"\b[a-zA-Z]{4,}\b", s) if w.lower() not in _STOPWORDS]
            if not toks:
                continue
            ans = random.choice(toks).lower()
        distract = [w for w in words if w != ans][:3]
        if len(distract) < 3:
            extra = [w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", s)]
            for w in extra:
                if w != ans and w not in distract:
                    distract.append(w)
                if len(distract) == 3:
                    break
        if len(distract) < 3:
            continue
        q = re.sub(rf"\b{re.escape(ans)}\b", "_____", s, flags=re.IGNORECASE)
        opts = [ans] + distract
        disp = [o.capitalize() for o in opts]
        random.shuffle(disp)
        correct_index = next(i for i,o in enumerate(disp) if o.lower() == ans.lower())
        mcqs.append({
            "question": q,
            "options": disp,
            "answer": chr(65+correct_index),
            "answer_text": ans
        })
        if len(mcqs) >= count:
            break
    return mcqs

def generate_important_questions(text, count=8):
    sents = _sentence_split(text)
    long_sents = [s for s in sents if len(s) > 80]
    long_sents.sort(key=len, reverse=True)
    qs = []
    for s in long_sents:
        qs.append("Explain: " + (s[:300] + ("..." if len(s) > 300 else "")))
        if len(qs) >= count:
            break
    if not qs:
        words = extract_frequent_words(text, count)
        qs = [f"Discuss the importance of '{w}'." for w in words]
    return qs[:count]

def generate_short_answers(text, count=8):
    sents = _sentence_split(text)
    keys = ["is", "are", "means", "refers to", "defined as", "consists of"]
    chosen = []
    for s in sents:
        if len(s.split()) < 40 and any(k in s.lower() for k in keys):
            chosen.append(s)
        if len(chosen) == count:
            break
    if len(chosen) < count:
        for s in sents:
            if s not in chosen:
                chosen.append(s)
                if len(chosen) == count:
                    break
    out = []
    for s in chosen:
        q = "What is: " + s.split(".")[0]
        out.append({"q": q, "a": s})
    return out[:count]

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Notes Generator (Free)", page_icon="📝", layout="wide")
st.title("AI Notes Generator — Free (ASCII-only PDF mode)")

uploaded = st.file_uploader("Upload a PDF or TXT", type=["pdf", "txt"])

summary_n = st.sidebar.slider("Summary Length", 3, 12, 6)
mcq_n = st.sidebar.slider("MCQs", 1, 12, 6)
imp_n = st.sidebar.slider("Important Questions", 1, 12, 6)
short_n = st.sidebar.slider("Short Answers", 1, 12, 6)
enable_pdf = st.sidebar.checkbox("Enable PDF", True)

if uploaded:
    text = extract_text(uploaded)
    if not text.strip():
        st.error("Empty or unreadable file.")
        st.stop()

    with st.expander("Preview"):
        st.write(text[:2000] + ("..." if len(text) > 2000 else ""))

    if st.button("Generate"):
        summary = simple_summary(text, summary_n)
        mcqs = generate_mcqs(text, mcq_n)
        imp = generate_important_questions(text, imp_n)
        short = generate_short_answers(text, short_n)
        probable = [f"Why is '{w}' important?" for w in extract_frequent_words(text, 8)]

        st.header("Summary")
        st.write(summary)

        st.header("MCQs")
        for i,m in enumerate(mcqs,1):
            st.write(f"**Q{i}.** {m['question']}")
            for oi,o in enumerate(m["options"]):
                st.write(f"- {chr(65+oi)}. {o}")
            st.write(f"**Answer:** {m['answer']}")

        st.header("Important Questions")
        for i,q in enumerate(imp,1):
            st.write(f"{i}. {q}")

        st.header("Short Answers")
        for i,p in enumerate(short,1):
            st.write(f"{i}. Q: {p['q']}")
            st.write(f"   A: {p['a']}")

        st.header("Probable Questions")
        for i,q in enumerate(probable,1):
            st.write(f"{i}. {q}")

        # TEXT Download
        out = (
            "=== SUMMARY ===\n" + summary + "\n\n" +
            "=== MCQs ===\n" +
            "\n".join([f"Q{i+1}. {m['question']}" for i,m in enumerate(mcqs)]) +
            "\n\n=== IMPORTANT QUESTIONS ===\n" +
            "\n".join(imp) +
            "\n\n=== SHORT ANSWERS ===\n" +
            "\n".join([f"{p['q']} → {p['a']}" for p in short]) +
            "\n\n=== PROBABLE ===\n" +
            "\n".join(probable)
        )

        st.download_button("Download TXT", out, file_name="notes.txt")

        # PDF (ASCII only — NO unicode errors ever)
        if enable_pdf:
            pdf_buf = io.BytesIO()
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=10)

            safe = _sanitize_ascii(out)

            for line in safe.splitlines():
                if not line.strip():
                    pdf.ln(4)
                    continue
                pdf.multi_cell(0, 5, line)

            pdf.output(pdf_buf)
            pdf_buf.seek(0)

            st.download_button(
                "Download PDF",
                pdf_buf,
                file_name="notes.pdf",
                mime="application/pdf"
            )
