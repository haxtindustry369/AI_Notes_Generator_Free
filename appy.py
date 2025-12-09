# app.py - Free Streamlit AI Notes Generator (no paid APIs, no spaCy)
import io, re, random
from fpdf import FPDF
import pdfplumber
import streamlit as st

# summarizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# nltk for simple POS and resources
import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn

# Ensure needed NLTK data (Streamlit will run this on server at first run)
nltk_packages = ["punkt", "averaged_perceptron_tagger", "wordnet", "omw-1.4"]
for pkg in nltk_packages:
    try:
        nltk.data.find(f"tokenizers/{pkg}") if pkg == "punkt" else nltk.data.find(f"taggers/{pkg}") if pkg == "averaged_perceptron_tagger" else nltk.data.find(f"corpora/{pkg}")
    except Exception:
        try:
            nltk.download(pkg)
        except Exception:
            pass  # if download fails on server, app may still run partially

st.set_page_config(page_title="AI Notes Generator (Free)", page_icon=":memo:", layout="wide")
st.title("AI Notes Generator — Free (no paid APIs)")

st.sidebar.header("Options")
summary_sentences = st.sidebar.slider("Summary sentences", 3, 12, 6)
num_mcq = st.sidebar.slider("Number of MCQs", 1, 12, 8)
num_imp_q = st.sidebar.slider("Important Questions", 1, 12, 8)
num_short_ans = st.sidebar.slider("Short Answers", 1, 12, 8)
enable_pdf = st.sidebar.checkbox("Enable PDF download", value=True)

uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

def extract_text_from_pdf(file_stream):
    text_parts = []
    with pdfplumber.open(file_stream) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    return "\n".join(text_parts)

def extract_text(file):
    if file.type == "application/pdf" or file.name.lower().endswith(".pdf"):
        return extract_text_from_pdf(file)
    else:
        raw = file.read()
        if isinstance(raw, bytes):
            try:
                return raw.decode("utf-8", errors="ignore")
            except Exception:
                return raw.decode("latin-1", errors="ignore")
        return str(raw)

def summarize_text(text, sentences_count=5):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(document=parser.document, sentences_count=sentences_count)
    return "\n".join([str(s) for s in summary])

def extract_nouns(text, topk=50):
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    nouns = [w for w,pos in tagged if pos.startswith('NN') and len(w)>2]
    freq = {}
    for n in nouns:
        freq[n.lower()] = freq.get(n.lower(),0) + 1
    sorted_n = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w,_ in sorted_n][:topk]

def make_mcqs_from_text(text, n_mcq=8):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    candidates = [s for s in sentences if len(s.split())>6]
    random.shuffle(candidates)
    freq_nouns = extract_nouns(text, topk=200)
    mcqs = []
    for s in candidates:
        # pick a noun in sentence
        words = word_tokenize(s)
        tagged = pos_tag(words)
        nouns_in_s = [w for w,p in tagged if p.startswith('NN') and len(w)>2]
        if not nouns_in_s:
            continue
        answer = random.choice(nouns_in_s)
        # build distractors: try wordnet synonyms/hypernyms, then frequent nouns
        distractors = []
        syns = []
        for syn in wn.synsets(answer):
            for lem in syn.lemmas():
                w = lem.name().replace('_',' ')
                if w.lower()!=answer.lower() and w not in distractors:
                    syns.append(w)
        for w in syns[:2]:
            distractors.append(w)
        for w in freq_nouns:
            if w.lower()!=answer.lower() and w not in distractors:
                distractors.append(w)
            if len(distractors)>=3:
                break
        if len(distractors)<3:
            # fallback: random words from sentence
            words_alpha = [t for t in words if t.isalpha() and t.lower()!=answer.lower()]
            random.shuffle(words_alpha)
            for w in words_alpha:
                if w not in distractors:
                    distractors.append(w)
                if len(distractors)>=3:
                    break
        if len(distractors)<3:
            continue
        q_text = s.replace(answer, "_____")
        options = [answer] + distractors[:3]
        random.shuffle(options)
        correct = chr(65 + options.index(answer))
        mcqs.append({
            "question": q_text.strip(),
            "options": options,
            "answer": correct,
            "answer_text": answer
        })
        if len(mcqs)>=n_mcq:
            break
    return mcqs

def make_important_questions(text, n=8):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    long_sorted = sorted([s.strip() for s in sentences if len(s.strip())>30], key=lambda s: len(s), reverse=True)
    qs = []
    for s in long_sorted[:n*2]:
        qs.append("Explain: " + (s[:320] + "..." if len(s)>320 else s))
        if len(qs)>=n:
            break
    return qs

def make_short_answers(text, n=8):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    keywords = ['is', 'are', 'means', 'refers to', 'defined as', 'consists of', 'used to', 'calculate']
    chosen = []
    for s in sentences:
        low = s.lower()
        if any(k in low for k in keywords) and len(s.split())<40:
            chosen.append(s.strip())
        if len(chosen)>=n:
            break
    if len(chosen)<n:
        for s in sentences:
            if s.strip() and s.strip() not in chosen:
                chosen.append(s.strip())
            if len(chosen)>=n:
                break
    pairs = []
    for s in chosen[:n]:
        q = f"What is: {s.split('.')[0][:120]}?"
        a = s.strip()
        pairs.append({"q":q,"a":a})
    return pairs

def most_likely_questions(text, n=8):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    headings = [l for l in lines if (l.isupper() and len(l.split())<10)]
    qs = []
    for h in headings[:n]:
        qs.append(f"Discuss: {h}")
    if len(qs)>=n:
        return qs[:n]
    freq_nouns = extract_nouns(text, topk=50)
    for noun in freq_nouns[:n]:
        qs.append(f"Why is '{noun}' important in this topic?")
        if len(qs)>=n:
            break
    if len(qs)<n:
        long_sents = sorted(re.split(r'(?<=[.!?])\s+', text), key=lambda s: len(s), reverse=True)
        for s in long_sents[:n]:
            qs.append("Discuss: " + (s[:150] + "..." if len(s)>150 else s))
            if len(qs)>=n:
                break
    return qs[:n]

# UI
if uploaded:
    st.info(f"Uploaded: {uploaded.name}")
    raw_text = extract_text(uploaded)
    if not raw_text.strip():
        st.error("No text extracted. If PDF is scanned, OCR is required (not included).")
        st.stop()

    with st.expander("Preview (first 2000 chars)"):
        st.write(raw_text[:2000] + ("..." if len(raw_text)>2000 else ""))

    if st.button("Generate (free)"):
        with st.spinner("Generating summary & questions..."):
            summary = summarize_text(raw_text, sentences_count=summary_sentences)
            mcqs = make_mcqs_from_text(raw_text, n_mcq=num_mcq)
            imp_qs = make_important_questions(raw_text, n=num_imp_q)
            short_ans = make_short_answers(raw_text, n=num_short_ans)
            likely_qs = most_likely_questions(raw_text, n=8)

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

        st.header("Short Answer Q&A")
        for i,p in enumerate(short_ans,1):
            st.write(f"{i}. Q: {p['q']}")
            st.write(f"   A: {p['a']}")

        st.header("Highly Likely Questions")
        for i,q in enumerate(likely_qs,1):
            st.write(f"{i}. {q}")

        # Prepare downloads
        full_txt = f"=== Summary ===\n{summary}\n\n=== MCQs ===\n"
        for i,m in enumerate(mcqs,1):
            full_txt += f"Q{i}. {m['question']}\n"
            for oi,opt in enumerate(m['options']):
                full_txt += f"  {chr(65+oi)}. {opt}\n"
            full_txt += f"Answer: {m['answer']} — {m['answer_text']}\n\n"
        full_txt += "\n=== Important Questions ===\n" + "\n".join(imp_qs) + "\n\n"
        full_txt += "=== Short Answers ===\n" + "\n".join([f"Q{i+1}. {p['q']}\nA: {p['a']}" for i,p in enumerate(short_ans)]) + "\n\n"
        full_txt += "=== Highly Likely Questions ===\n" + "\n".join(likely_qs)

        st.download_button("Download .txt", full_txt, file_name=f"{uploaded.name}_notes.txt", mime="text/plain")

        if enable_pdf:
            buf = io.BytesIO()
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=12)
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0,6,txt=f"AI Notes (Free) for {uploaded.name}\n\n")
            pdf.multi_cell(0,6,txt="Summary:\n"+summary+"\n\n")
            pdf.multi_cell(0,6,txt="MCQs:\n")
            for i,m in enumerate(mcqs,1):
                pdf.multi_cell(0,6,txt=f"{i}. {m['question']}")
                for oi,opt in enumerate(m['options']):
                    pdf.multi_cell(0,6,txt=f"    {chr(65+oi)}. {opt}")
                pdf.multi_cell(0,6,txt=f"    Answer: {m['answer']} — {m['answer_text']}\n")
            pdf.multi_cell(0,6,txt="\nImportant Questions:\n" + "\n".join(imp_qs) + "\n\n")
            pdf.multi_cell(0,6,txt="Short Answer Q&A:\n" + "\n".join([f"Q{i+1}. {p['q']}\nA: {p['a']}" for i,p in enumerate(short_ans)]) + "\n\n")
            pdf.multi_cell(0,6,txt="Highly Likely Questions:\n" + "\n".join(likely_qs))
            pdf.output(buf)
            buf.seek(0)
            st.download_button("Download .pdf", buf, file_name=f"{uploaded.name}_notes.pdf", mime="application/pdf")
else:
    st.info("Upload a PDF or TXT file to start (free version).")
