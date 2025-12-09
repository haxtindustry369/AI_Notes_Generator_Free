# app.py - Free Streamlit AI Notes Generator (no paid APIs, no spaCy)
import io, re, random
from fpdf import FPDF
import pdfplumber
import streamlit as st

# -------------------------------------------------------------------
# NLTK SETUP (fully safe and compatible with Streamlit Cloud)
# -------------------------------------------------------------------
import nltk

nltk_packages = ["punkt", "averaged_perceptron_tagger", "wordnet", "omw-1.4"]
for pkg in nltk_packages:
    try:
        if pkg == "punkt":
            nltk.data.find("tokenizers/punkt")
        elif pkg == "averaged_perceptron_tagger":
            nltk.data.find("taggers/averaged_perceptron_tagger")
        else:
            nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        try:
            nltk.download(pkg)
        except:
            pass  # Streamlit may block sometimes; fallback logic handles it

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn

# -------------------------------------------------------------------
# SUMMARIZER (Sumy + robust fallback)
# -------------------------------------------------------------------
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def simple_summary(text, sentences_count=5):
    """Fallback summary: first N sentences."""
    sents = re.split(r'(?<=[.!?])\s+', text)
    return "\n".join(sents[:sentences_count])

def summarize_text(text, sentences_count=5):
    """Robust summarizer with automatic retry and fallback."""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = LexRankSummarizer()
        summary = summarizer(parser.document, sentences_count)
        return "\n".join(str(s) for s in summary)
    except:
        # Retry after punkt download
        try:
            nltk.download("punkt")
            parser = PlaintextParser.from_string(text, Tokenizer("english"))
            summarizer = LexRankSummarizer()
            summary = summarizer(parser.document, sentences_count)
            return "\n".join(str(s) for s in summary)
        except:
            return simple_summary(text, sentences_count)

# -------------------------------------------------------------------
# TEXT EXTRACTION
# -------------------------------------------------------------------
def extract_text_from_pdf(file_stream):
    text_parts = []
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)

def extract_text(file):
    if file.type == "application/pdf" or file.name.lower().endswith(".pdf"):
        return extract_text_from_pdf(file)
    else:
        raw = file.read()
        try:
            return raw.decode("utf-8", errors="ignore")
        except:
            return raw.decode("latin-1", errors="ignore")

# -------------------------------------------------------------------
# QUESTION GENERATORS
# -------------------------------------------------------------------
def extract_nouns(text, topk=50):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    nouns = [w for w,p in tagged if p.startswith("NN") and len(w)>2]
    freq = {}
    for n in nouns:
        freq[n.lower()] = freq.get(n.lower(),0)+1
    sorted_n = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w,_ in sorted_n][:topk]

def make_mcqs(text, count=8):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    candidates = [s for s in sentences if len(s.split())>6]
    random.shuffle(candidates)

    nouns = extract_nouns(text, topk=200)
    mcqs = []

    for s in candidates:
        words = word_tokenize(s)
        tagged = pos_tag(words)
        nouns_in_s = [w for w,p in tagged if p.startswith("NN")]

        if not nouns_in_s:
            continue

        answer = random.choice(nouns_in_s)

        distractors = []

        # Try WordNet synonyms
        for syn in wn.synsets(answer):
            for lemma in syn.lemmas():
                w = lemma.name().replace("_", " ")
                if w.lower() != answer.lower():
                    distractors.append(w)
        distractors = list(dict.fromkeys(distractors))[:2]  # remove duplicates

        # Add frequent nouns
        for n in nouns:
            if len(distractors) >= 3:
                break
            if n.lower() != answer.lower():
                distractors.append(n)

        # Fallback random distractors
        if len(distractors) < 3:
            w2 = [w for w in words if w.isalpha() and w.lower()!=answer.lower()]
            random.shuffle(w2)
            for w in w2:
                distractors.append(w)
                if len(distractors)>=3:
                    break

        if len(distractors) < 3:
            continue

        # Mask answer
        q_text = s.replace(answer, "_____")

        options = [answer] + distractors[:3]
        random.shuffle(options)
        correct = chr(65 + options.index(answer))

        mcqs.append({
            "question": q_text,
            "options": options,
            "answer": correct,
            "answer_text": answer
        })

        if len(mcqs) >= count:
            break

    return mcqs

def make_important_questions(text, count=8):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    long_sentences = sorted(
        [s for s in sentences if len(s.strip())>30],
        key=lambda s: len(s),
        reverse=True
    )
    qs = []
    for s in long_sentences:
        qs.append("Explain: " + s[:300])
        if len(qs) >= count:
            break
    return qs

def make_short_answers(text, count=8):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    keywords = ['is', 'are', 'means', 'refers to', 'defined as']

    chosen = []
    for s in sentences:
        if len(chosen) >= count:
            break
        if any(k in s.lower() for k in keywords):
            chosen.append(s.strip())

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

def make_probable_questions(text, count=8):
    nouns = extract_nouns(text, topk=20)
    qs = []
    for n in nouns[:count]:
        qs.append(f"Why is '{n}' important in this topic?")
    return qs

# -------------------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------------------
st.set_page_config(page_title="AI Notes Generator Free", page_icon="📘")
st.title("📘 AI Notes Generator — FREE Version")

st.sidebar.header("Options")
summary_len = st.sidebar.slider("Summary sentences:", 3, 12, 6)
mcq_count = st.sidebar.slider("Number of MCQs:", 1, 12, 6)
impq_count = st.sidebar.slider("Important Questions:", 1, 12, 6)
sa_count = st.sidebar.slider("Short Answers:", 1, 12, 6)
enable_pdf = st.sidebar.checkbox("Enable PDF download", True)

uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

if uploaded:
    raw_text = extract_text(uploaded)

    if not raw_text.strip():
        st.error("No extractable text found. Scanned PDFs are not supported (need OCR).")
        st.stop()

    with st.expander("Preview text"):
        st.write(raw_text[:2000] + "...")

    if st.button("Generate (FREE)"):
        with st.spinner("Processing..."):
            summary = summarize_text(raw_text, summary_len)
            mcqs = make_mcqs(raw_text, mcq_count)
            impqs = make_important_questions(raw_text, impq_count)
            shortans = make_short_answers(raw_text, sa_count)
            probable = make_probable_questions(raw_text, 6)

        st.header("Summary")
        st.write(summary)

        st.header("MCQs")
        for i,m in enumerate(mcqs,1):
            st.write(f"**Q{i}.** {m['question']}")
            for oi,opt in enumerate(m['options']):
                st.write(f"- {chr(65+oi)}. {opt}")
            st.write(f"**Answer:** {m['answer']} ({m['answer_text']})")

        st.header("Important Questions")
        for i,q in enumerate(impqs,1):
            st.write(f"{i}. {q}")

        st.header("Short Answer Questions")
        for i,p in enumerate(shortans,1):
            st.write(f"{i}. Q: {p['q']}")
            st.write(f"   A: {p['a']}")

        st.header("Most Probable Questions")
        for i,q in enumerate(probable,1):
            st.write(f"{i}. {q}")

        # ---------------------------
        # DOWNLOAD AS TEXT
        # ---------------------------
        txt = f"SUMMARY:\n{summary}\n\nMCQs:\n"
        for m in mcqs:
            txt += m['question'] + "\n"
            for opt in m['options']:
                txt += f"- {opt}\n"
            txt += f"Answer: {m['answer_text']}\n\n"

        st.download_button("Download Notes (.txt)", txt, file_name="notes.txt")

        # ---------------------------
        # DOWNLOAD AS PDF
        # ---------------------------
        if enable_pdf:
            buf = io.BytesIO()
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=10)
            pdf.set_font("Arial", size=12)

            pdf.multi_cell(0, 6, f"AI Notes Generated (FREE)\n\nSUMMARY:\n{summary}\n\nMCQs:\n")
            for m in mcqs:
                pdf.multi_cell(0, 6, f"{m['question']}")
                for opt in m['options']:
                    pdf.multi_cell(0, 6, f"- {opt}")
                pdf.multi_cell(0, 6, f"Answer: {m['answer_text']}\n")

            pdf.output(buf)
            buf.seek(0)
            st.download_button("Download Notes (.pdf)", buf, file_name="notes.pdf", mime="application/pdf")
else:
    st.info("Upload a PDF or TXT file to begin.")

