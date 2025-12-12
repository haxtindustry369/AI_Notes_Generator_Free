# app.py - Streamlit AI Notes Generator with OCR.space fallback (works on Streamlit Cloud)
import io
import os
import re
import random
import collections
import unicodedata
import requests
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
    def _chunk(match):
        s = match.group(0)
        return " ".join(s[i:i+max_len] for i in range(0, len(s), max_len))
    return re.sub(r'\S{' + str(max_len) + r',}', _chunk, text)

def _sanitize_ascii(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii", "ignore")
    return _break_long(text, max_len=30)

# -------------------------
# OCR.space helper
# -------------------------
def ocr_space_api(file_bytes: bytes, api_key: str, language: str = "eng", is_pdf: bool = False) -> str:
    """
    Send file bytes to OCR.space and return extracted text (or empty string on failure).
    """
    url = "https://api.ocr.space/parse/image"
    files = {
        "file": ("upload.pdf" if is_pdf else "upload.png", file_bytes)
    }
    data = {
        "apikey": api_key,
        "language": language,
        "isOverlayRequired": False,
        "OCREngine": 2
    }
    try:
        resp = requests.post(url, files=files, data=data, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        if result.get("IsErroredOnProcessing"):
            return ""
        parsed = result.get("ParsedResults")
        if not parsed:
            return ""
        text = "\n".join(p.get("ParsedText", "") for p in parsed)
        return text
    except Exception:
        return ""

# -------------------------
# Extraction & summarization
# -------------------------
def extract_text_from_pdf(file_stream):
    text_parts = []
    try:
        with pdfplumber.open(file_stream) as pdf:
            for page in pdf.pages:
                text_parts.append(page.extract_text() or "")
    except Exception:
        # fallback to reading raw bytes
        try:
            file_stream.seek(0)
            raw = file_stream.read()
            text_parts.append(raw.decode("utf-8", errors="ignore"))
        except Exception:
            pass
    return "\n".join(text_parts)

def extract_text(uploaded_file):
    # return tuple (extracted_text, used_ocr_bool)
    uploaded_file.seek(0)
    if uploaded_file.name.lower().endswith(".pdf") or uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        raw = uploaded_file.read()
        if isinstance(raw, bytes):
            try:
                text = raw.decode("utf-8", errors="ignore")
            except:
                text = raw.decode("latin-1", errors="ignore")
        else:
            text = str(raw)

    used_ocr = False
    if not text or not text.strip():
        # Try OCR via OCR.space if API key present in secrets
        api_key = st.secrets.get("OCR_SPACE_API_KEY") if "OCR_SPACE_API_KEY" in st.secrets else None
        if api_key:
            uploaded_file.seek(0)
            file_bytes = uploaded_file.read()
            is_pdf = uploaded_file.name.lower().endswith(".pdf")
            ocr_text = ocr_space_api(file_bytes, api_key, language="eng", is_pdf=is_pdf)
            if ocr_text and ocr_text.strip():
                text = ocr_text
                used_ocr = True
    return text, used_ocr

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
# MCQ / question generation
# -------------------------
def generate_mcqs(text, count=6):
    sentences = _sentence_split(text)
    candidates = [s for s in sentences if len(s.split()) > 6]
    random.shuffle(candidates)
    freq_words = extract_frequent_words(text, topk=200)
    mcqs = []
    for s in candidates:
        s_lower = s.lower()
        answers_in_sent = [w for w in freq_words if re.search(r"\b" + re.escape(w) + r"\b", s_lower)]
        if answers_in_sent:
            answer = random.choice(answers_in_sent)
        else:
            words_in_sent = re.findall(r"\b[a-zA-Z]{4,}\b", s)
            words_in_sent = [w for w in words_in_sent if w.lower() not in _STOPWORDS]
            if not words_in_sent:
                continue
            answer = random.choice(words_in_sent).lower()

        distractors = [w for w in freq_words if w != answer]
        random.shuffle(distractors)
        distractors = distractors[:3]
        if len(distractors) < 3:
            extra = [w.lower() for w in re.findall(r"\b[a-zA-Z]{4,}\b", s) if w.lower() != answer and w.lower() not in _STOPWORDS]
            random.shuffle(extra)
            for w in extra:
                if len(distractors) >= 3:
                    break
                if w not in distractors:
                    distractors.append(w)

        if len(distractors) < 3:
            continue

        q_text = re.sub(re.escape(answer), "_____", s, flags=re.IGNORECASE)
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
    long_sentences = sorted([s for s in sentences if len(s) > 80], key=lambda s: len(s), reverse=True)
    qs = []
    for s in long_sentences[:count]:
        qs.append("Explain: " + (s[:300] + ("..." if len(s) > 300 else "")))
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
            if s.strip():
                chosen.append(s.strip())
            if len(chosen) >= count:
                break
    qa = []
    for s in chosen[:count]:
        q = "What is: " + s[:60] + "?"
        qa.append({"q": q, "a": s})
    return qa

# -------------------------
# Defensive PDF writing helpers
# -------------------------
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

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="AI Notes Generator (OCR)", page_icon="📝", layout="wide")
st.title("AI Notes Generator — with OCR (uses OCR.space when needed)")

st.sidebar.header("Options")
summary_sentences = st.sidebar.slider("Summary sentences", 3, 12, 6)
mcq_count = st.sidebar.slider("Number of MCQs", 1, 12, 6)
imp_q_count = st.sidebar.slider("Important Questions", 1, 12, 6)
sa_count = st.sidebar.slider("Short Answers", 1, 12, 6)
enable_pdf = st.sidebar.checkbox("Enable PDF download", True)

uploaded = st.file_uploader("Upload a PDF or TXT file (OCR used if no extractable text)", type=["pdf", "txt", "png", "jpg", "jpeg"])

if uploaded:
    st.info(f"Uploaded: {uploaded.name}")
    raw_text, used_ocr = extract_text(uploaded)
    if used_ocr:
        st.success("Text extracted using OCR (OCR.space).")
    else:
        st.info("Text extracted using pdfplumber/plain text.")
    if not raw_text or not raw_text.strip():
        st.error("No extractable text found and OCR either not configured or failed.")
        st.stop()

    with st.expander("Preview (first 2000 chars)"):
        st.write(raw_text[:2000] + ("..." if len(raw_text) > 2000 else ""))

    if st.button("Generate"):
        summary = simple_summary(raw_text, summary_sentences)
        mcqs = generate_mcqs(raw_text, mcq_count)
        imp_qs = generate_important_questions(raw_text, imp_q_count)
        short_ans = generate_short_answers(raw_text, sa_count)
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

        # download text
        result_txt = f"=== SUMMARY ===\n{summary}\n\n=== MCQS ===\n"
        for i,m in enumerate(mcqs,1):
            result_txt += f"Q{i}. {m['question']}\n"
            for oi,opt in enumerate(m['options']):
                result_txt += f"  {chr(65+oi)}. {opt}\n"
            result_txt += f"Answer: {m['answer_text']}\n\n"
        result_txt += "\n=== IMPORTANT QUESTIONS ===\n" + "\n".join(imp_qs) + "\n\n"
        result_txt += "=== SHORT ANSWERS ===\n" + "\n".join([f"Q{i+1}. {p['q']}\nA: {p['a']}" for i,p in enumerate(short_ans)]) + "\n\n"
        result_txt += "=== MOST PROBABLE ===\n" + "\n".join(probable)

        st.download_button("Download TXT", result_txt, file_name=f"{uploaded.name}_notes.txt", mime="text/plain")

        # build PDF blocks with fallback to include original text if nothing generated
        safe_text_blocks = []
        safe_text_blocks.append("AI Notes (Generated)\n")

        use_raw_fallback = False
        if (not summary or not summary.strip()) and (not mcqs) and (not imp_qs) and (not short_ans):
            use_raw_fallback = True

        if use_raw_fallback:
            fallback_chunk = raw_text[:4000] if len(raw_text) > 4000 else raw_text
            safe_text_blocks.append("EXTRACTED TEXT (fallback):\n")
            safe_text_blocks.append(_sanitize_ascii(fallback_chunk))
        else:
            safe_text_blocks.append("SUMMARY:\n" + _sanitize_ascii(summary) + "\n\n")
            safe_text_blocks.append("MCQS:\n")
            for idx, m in enumerate(mcqs, 1):
                safe_text_blocks.append(_sanitize_ascii(f"{idx}. {m['question']}"))
                for oi, opt in enumerate(m['options']):
                    safe_text_blocks.append(_sanitize_ascii(f"   {chr(65+oi)}. {opt}"))
                safe_text_blocks.append(_sanitize_ascii(f"   Answer: {m['answer_text']}\n"))
            safe_text_blocks.append("\nImportant Questions:\n" + _sanitize_ascii("\n".join(imp_qs)) + "\n\n")
            safe_text_blocks.append("Short Answers:\n")
            for i, p in enumerate(short_ans, 1):
                safe_text_blocks.append(_sanitize_ascii(f"Q{i}. {p['q']}"))
                safe_text_blocks.append(_sanitize_ascii(f"A: {p['a']}\n"))
            safe_text_blocks.append("Most Probable Questions:\n" + _sanitize_ascii("\n".join(probable)))

        # PDF generation (ASCII-safe)
        if enable_pdf:
            pdf_buf = io.BytesIO()
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=10)
            pdf.set_left_margin(10)
            pdf.set_right_margin(10)
            pdf.set_auto_page_break(auto=True, margin=12)

            for block in safe_text_blocks:
                for line in block.splitlines():
                    if not line.strip():
                        pdf.ln(4)
                        continue
                    for part in _split_to_chunks(line, max_chars=150):
                        _write_defensively(pdf, part, h=5, max_chunk=120)

            pdf.output(pdf_buf)
            pdf_buf.seek(0)
            st.download_button("Download PDF", pdf_buf, file_name=f"{uploaded.name}_notes.pdf", mime="application/pdf")
else:
    st.info("Upload a PDF, image, or TXT file. If PDF is scanned, OCR.space will be used when configured.")
