# app.py - Minimal, robust free notes generator with unicode-safe PDF fallback
import io
import os
import re
import random
import collections
from fpdf import FPDF
import pdfplumber
import streamlit as st
import unicodedata

# -------------------------
# Helper: safe-for-pdf function (always available)
# -------------------------
def _make_safe_for_pdf(s: str, allow_unicode: bool) -> str:
    """
    If allow_unicode is True, return string unchanged.
    Otherwise normalize and strip non-ascii characters so FPDF (ASCII) won't crash.
    """
    if allow_unicode:
        return s
    normalized = unicodedata.normalize("NFKD", s)
    ascii_bytes = normalized.encode("ascii", "ignore")
    return ascii_bytes.decode("ascii", "ignore")
# helper: insert spaces into very long words so FPDF can wrap them
def _break_long_unbroken_sequences(text: str, max_len: int = 60) -> str:
    """
    Insert a space every `max_len` characters inside any long sequence of
    non-space characters to avoid FPDF "no horizontal space" errors.
    """
    # This will replace sequences of non-space characters longer than max_len
    # with chunks separated by a space (so FPDF can break).
    def _chunk(match):
        s = match.group(0)
        return " ".join(s[i:i+max_len] for i in range(0, len(s), max_len))
    return re.sub(r'\S{' + str(max_len) + r',}', _chunk, text)

# -------------------------
# Simple stopwords + tokenizers (no external NLP libs)
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

# -------------------------
# Extraction and summarization (lightweight)
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
    if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    else:
        raw = uploaded_file.read()
        if isinstance(raw, bytes):
            try:
                return raw.decode("utf-8", errors="ignore")
            except:
                return raw.decode("latin-1", errors="ignore")
        return str(raw)

def simple_summary(text, sentence_count=5):
    sents = _sentence_split(text)
    if not sents:
        return ""
    words = _simple_tokenize_words(text)
    freq = collections.Counter(w for w in words if w not in _STOPWORDS)
    if not freq:
        return "\n".join(sents[:sentence_count])
    sent_scores = []
    for s in sents:
        tokens = _simple_tokenize_words(s)
        score = sum(freq.get(t,0) for t in tokens)
        sent_scores.append((s, score, len(tokens)))
    sent_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    chosen = [s for s,_,_ in sent_scores[:sentence_count]]
    chosen_ordered = [s for s in sents if s in chosen]
    return "\n".join(chosen_ordered)

def extract_frequent_words(text, topk=50):
    tokens = _simple_tokenize_words(text)
    freq = collections.Counter(t for t in tokens if t not in _STOPWORDS)
    return [w for w,_ in freq.most_common(topk)]

# -------------------------
# MCQ / question generators
# -------------------------
def generate_mcqs(text, count=6):
    sentences = _sentence_split(text)
    candidates = [s for s in sentences if len(s.split()) > 6]
    random.shuffle(candidates)
    freq_words = extract_frequent_words(text, topk=200)
    mcqs = []
    for s in candidates:
        s_lower = s.lower()
        possible_answers = [w for w in freq_words if re.search(r"\b" + re.escape(w) + r"\b", s_lower)]
        if possible_answers:
            answer = random.choice(possible_answers)
        else:
            words = re.findall(r"\b[a-zA-Z]{4,}\b", s)
            words = [w for w in words if w.lower() not in _STOPWORDS]
            if not words:
                continue
            answer = random.choice(words).lower()
        distractors = [w for w in freq_words if w != answer]
        random.shuffle(distractors)
        distractors = distractors[:3]
        if len(distractors) < 3:
            extras = [w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", s) if w.lower() != answer and w.lower() not in _STOPWORDS]
            random.shuffle(extras)
            for e in extras:
                if e not in distractors:
                    distractors.append(e)
                if len(distractors) >= 3:
                    break
        if len(distractors) < 3:
            continue
        pattern = re.compile(re.escape(answer), re.IGNORECASE)
        q_text = pattern.sub("_____", s, count=1)
        options = [answer] + distractors[:3]
        options_display = [opt.capitalize() for opt in options]
        random.shuffle(options_display)
        correct_index = next(i for i,opt in enumerate(options_display) if opt.lower() == answer.lower())
        correct_letter = chr(65 + correct_index)
        mcqs.append({
            "question": q_text,
            "options": options_display,
            "answer": correct_letter,
            "answer_text": answer
        })
        if len(mcqs) >= count:
            break
    return mcqs

def generate_important_questions(text, count=8):
    sentences = _sentence_split(text)
    long_sorted = sorted([s for s in sentences if len(s) > 80], key=lambda x: len(x), reverse=True)
    qs = []
    for s in long_sorted:
        qs.append("Explain: " + (s[:300] + ("..." if len(s) > 300 else "")))
        if len(qs) >= count:
            break
    if not qs:
        freq = extract_frequent_words(text, topk=count)
        for w in freq:
            qs.append(f"Discuss the importance of '{w}' in this topic.")
            if len(qs) >= count:
                break
    return qs[:count]

def generate_short_answers(text, count=8):
    sentences = _sentence_split(text)
    keywords = ['is', 'are', 'means', 'refers to', 'defined as', 'consists of', 'used to', 'calculate']
    chosen = []
    for s in sentences:
        low = s.lower()
        if any(k in low for k in keywords) and len(s.split()) < 40:
            chosen.append(s.strip())
        if len(chosen) >= count:
            break
    if len(chosen) < count:
        for s in sentences:
            if s.strip() and s.strip() not in chosen:
                chosen.append(s.strip())
            if len(chosen) >= count:
                break
    pairs = []
    for s in chosen[:count]:
        q = f"What is: {s.split('.')[0][:100]}?"
        pairs.append({"q": q, "a": s})
    return pairs

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Notes Generator (Free)", page_icon=":memo:", layout="wide")
st.title("AI Notes Generator — Free (no external models)")

st.sidebar.header("Options")
summary_sentences = st.sidebar.slider("Summary sentences", 3, 12, 6)
num_mcq = st.sidebar.slider("Number of MCQs", 1, 12, 6)
num_imp_q = st.sidebar.slider("Important Questions", 1, 12, 6)
num_short_ans = st.sidebar.slider("Short Answers", 1, 12, 6)
enable_pdf = st.sidebar.checkbox("Enable PDF download", value=True)

uploaded = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded:
    st.info(f"Uploaded: {uploaded.name}")
    raw_text = extract_text(uploaded)
    if not raw_text or not raw_text.strip():
        st.error("Could not extract text. If the PDF is scanned (images), OCR is required and not included here.")
        st.stop()

    with st.expander("Preview (first 2000 chars)"):
        st.write(raw_text[:2000] + ("..." if len(raw_text) > 2000 else ""))

    if st.button("Generate (FREE)"):
        with st.spinner("Generating summary and questions..."):
            summary = simple_summary(raw_text, summary_sentences)
            mcqs = generate_mcqs(raw_text, num_mcq)
            imp_qs = generate_important_questions(raw_text, num_imp_q)
            short_ans = generate_short_answers(raw_text, num_short_ans)
            probable = [f"Why is '{w}' important?" for w in extract_frequent_words(raw_text, topk=8)]

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
            st.write("No MCQs could be generated from this text.")

        st.header("Important Questions")
        for i,q in enumerate(imp_qs,1):
            st.write(f"{i}. {q}")

        st.header("Short Answer Q&A")
        for i,p in enumerate(short_ans,1):
            st.write(f"{i}. Q: {p['q']}")
            st.write(f"   A: {p['a']}")

        st.header("Most Probable Questions")
        for i,q in enumerate(probable,1):
            st.write(f"{i}. {q}")

        # -------------------------
        # Download as TXT
        # -------------------------
        result_txt = f"=== SUMMARY ===\n{summary}\n\n=== MCQS ===\n"
        for i,m in enumerate(mcqs,1):
            result_txt += f"Q{i}. {m['question']}\n"
            for oi,opt in enumerate(m['options']):
                result_txt += f"  {chr(65+oi)}. {opt}\n"
            result_txt += f"Answer: {m['answer_text']}\n\n"
        result_txt += "\n=== IMPORTANT QUESTIONS ===\n" + "\n".join(imp_qs) + "\n\n"
        result_txt += "=== SHORT ANSWERS ===\n" + "\n".join([f"Q{i+1}. {p['q']}\nA: {p['a']}" for i,p in enumerate(short_ans)]) + "\n\n"
        result_txt += "=== MOST PROBABLE ===\n" + "\n".join(probable)

        st.download_button("Download .txt", result_txt, file_name=f"{uploaded.name}_notes.txt", mime="text/plain")

        # -------------------------
        # PDF generation (unicode-safe with fallback)
        # -------------------------
        if enable_pdf:
            buf = io.BytesIO()
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=12)
            pdf.add_page()

            # find font file
            base_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
            font_filename = "DejaVuSans.ttf"
            font_path = os.path.join(base_dir, font_filename)

            font_loaded = False
            try:
                if os.path.exists(font_path):
                    pdf.add_font("DejaVu", "", font_path, uni=True)
                    pdf.set_font("DejaVu", size=12)
                    font_loaded = True
                    st.info(f"Using font: {font_path}")
                else:
                    pdf.add_font("DejaVu", "", font_filename, uni=True)
                    pdf.set_font("DejaVu", size=12)
                    font_loaded = True
                    st.info(f"Using font by name: {font_filename}")
            except Exception:
                st.warning("Could not load Unicode font. PDF will fallback to ASCII-only text.")
                font_loaded = False

            try:
                pdf.multi_cell(0, 6, txt=_make_safe_for_pdf(f"AI Notes (FREE)\n\nSUMMARY:\n{summary}\n\n", font_loaded))
                pdf.multi_cell(0, 6, txt=_make_safe_for_pdf("MCQs:\n", font_loaded))
                for i, m in enumerate(mcqs, 1):
                    pdf.multi_cell(0, 6, txt=_make_safe_for_pdf(f"{i}. {m['question']}", font_loaded))
                    for oi, opt in enumerate(m['options']):
                        pdf.multi_cell(0, 6, txt=_make_safe_for_pdf(f"   {chr(65+oi)}. {opt}", font_loaded))
                    pdf.multi_cell(0, 6, txt=_make_safe_for_pdf(f"   Answer: {m['answer_text']}\n", font_loaded))

                pdf.multi_cell(0, 6, txt=_make_safe_for_pdf("\nImportant Questions:\n" + "\n".join(imp_qs) + "\n\n", font_loaded))

                pdf.multi_cell(0, 6, txt=_make_safe_for_pdf("Short Answers:\n", font_loaded))
                for i, p in enumerate(short_ans, 1):
                    pdf.multi_cell(0, 6, txt=_make_safe_for_pdf(f"Q{i}. {p['q']}", font_loaded))
                    pdf.multi_cell(0, 6, txt=_make_safe_for_pdf(f"A: {p['a']}\n", font_loaded))

                pdf.multi_cell(0, 6, txt=_make_safe_for_pdf("Most Probable Questions:\n" + "\n".join(probable), font_loaded))

                pdf.output(buf)
                buf.seek(0)
                st.download_button(
                    "Download .pdf",
                    buf,
                    file_name=f"{uploaded.name}_notes.pdf",
                    mime="application/pdf"
                )
            except Exception:
                # fallback ASCII-only pdf (robust)
                st.error("PDF generation failed with Unicode font, creating simple ASCII PDF as fallback.")
                fallback_buf = io.BytesIO()
                fallback_pdf = FPDF()
                fallback_pdf.add_page()

                # use a slightly smaller font for fallback to reduce space issues
                fallback_pdf.set_font("Arial", size=10)

                # prepare safe text: normalize and remove non-ascii
                safe_text = _make_safe_for_pdf(f"AI Notes (FREE)\n\nSUMMARY:\n{summary}\n\n", False)
                # ensure extremely long tokens are splitted so FPDF can wrap lines
                safe_text = _break_long_unbroken_sequences(safe_text, max_len=60)

                # write to pdf line-by-line (multi_cell can still wrap)
                for line in safe_text.splitlines():
                    # ensure each line itself isn't an empty large chunk
                    if not line.strip():
                       fallback_pdf.ln(4)
                       continue
                    fallback_pdf.multi_cell(0, 6, txt=line)

                fallback_pdf.output(fallback_buf)
                fallback_buf.seek(0)
                st.download_button(
                    "Download fallback .pdf",
                    fallback_buf,
                    file_name=f"{uploaded.name}_notes_ascii.pdf",
                    mime="application/pdf"
                )

else:
    st.info("Upload a PDF or TXT file to start (free).")

