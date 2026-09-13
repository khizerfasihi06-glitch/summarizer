import os
from datetime import datetime

import streamlit as st
import requests
from bs4 import BeautifulSoup
import PyPDF2
import docx
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="AI notes Summarizer", page_icon="📝", layout="wide")


def extract_text_from_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"
    return text.strip()


def extract_text_from_docx(file) -> str:
    document = docx.Document(file)
    return "\n".join(p.text for p in document.paragraphs).strip()


def extract_text_from_txt(file):
    return file.read().decode("utf-8", errors="ignore").strip()


def extract_text_from_url(url: str) -> str:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; NotesSummarizer/1.0)"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    paragraph = soup.find_all("p")
    text = "\n".join(p.get_text(strip=True) for p in paragraph if p.get_text(strip=True))
    if not text.strip():
        text = soup.get_text(separator="\n", strip=True)
    return text.strip()


def load_input_text(source_type: str, **kwargs) -> str:
    if source_type == "text":
        return kwargs["text"].strip()

    if source_type == "file":
        uploaded = kwargs["file"]
        name = uploaded.name.lower()
        if name.endswith(".pdf"):
            return extract_text_from_pdf(uploaded)
        elif name.endswith(".docx"):
            return extract_text_from_docx(uploaded)
        elif name.endswith(".txt"):
            return extract_text_from_txt(uploaded)
        else:
            raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

    if source_type == "url":
        return extract_text_from_url(kwargs["url"])

    return ""


# Removed Mistral to exclusively leverage Llama architecture on Groq
MODEL = ['openai/gpt-oss-120b', 'openai/gpt-oss-20b']

length_guidance = {
    "short": "keep the summary very tight 3-5 sentence max.",
    "medium": "Keep the summary to one solid paragraph (6-9 sentences)",
    "detailed": "Write a thorough multi-paragraph summary covering all major sub-topics",
}

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert note-taking assistant. Given raw notes, a transcript, or an "
            "article, produce a structured breakdown with exactly these three Markdown sections:\n\n"
            "## Summary\n"
            "## Key Points\n"
            "## Action Items\n\n"
            "Key Points should be a bulleted list of the most important facts or ideas. "
            "Action Items should be a bulleted list of any tasks, follow-ups, deadlines, decisions, "
            "or owners mentioned in the text. If there genuinely are none, write exactly: "
            "'No action items identified.' under that section.\n\n"
            "Only use information present in the text — never invent details. {length_guidance}"
        ),
        ("human", "Note to process: \n\n{text}"),
    ]
)


QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert exam-preparation assistant for university students, familiar with the "
            "typical exam style used at Sir Syed University (SSUET) — a mix of Multiple Choice Questions, "
            "Short Questions, and Long/descriptive Questions. Given the study material below, generate "
            "likely exam questions with model answers, formatted in Markdown with exactly these three "
            "sections, in this order:\n\n"
            "## Multiple Choice Questions\n"
            "Numbered MCQs, each with four options (a-d), followed immediately by 'Answer: <letter>' "
            "and a one-line explanation.\n\n"
            "## Short Questions\n"
            "Numbered short-answer questions (2-3 marks style), each followed by a concise 2-4 sentence "
            "model answer.\n\n"
            "## Long Questions\n"
            "Numbered descriptive/essay-style questions (10 marks style), each followed by a structured, "
            "well-organized model answer covering the key points from the material.\n\n"
            "Generate {num_mcq} MCQs, {num_short} short questions, and {num_long} long questions. "
            "Base every question and answer strictly on the material provided — never invent facts not "
            "present in or reasonably inferable from the text. Prioritize the concepts, definitions, "
            "processes, and comparisons most likely to be tested in a university exam."
        ),
        ("human", "Study material: \n\n{text}"),
    ]
)


def get_llm(api_key: str, model: str) -> ChatGroq:
    return ChatGroq(groq_api_key=api_key, model_name=model, temperature=0.3)


def summarize(llm: ChatGroq, text: str, length: str) -> str:
    CHUNK_LIMIT_CHARS = 15000
    chain = SUMMARY_PROMPT | llm | StrOutputParser()

    if len(text) <= 25000:
        return chain.invoke({"text": text, "length_guidance": length_guidance[length]})

    st.info("⚠️ This is a large document. Processing via a multi-stage summary pipeline...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_LIMIT_CHARS,
        chunk_overlap=1500
    )
    chunks = text_splitter.split_text(text)

    chunk_summaries = []
    progress_bar = st.progress(0.0)
    status = st.empty()

    for i, chunk in enumerate(chunks):
        status.write(f"Processing chunk {i + 1} of {len(chunks)}...")
        partial_res = chain.invoke({
            "text": chunk,
            "length_guidance": "Keep this component snippet dense and concise."
        })
        chunk_summaries.append(partial_res)
        progress_bar.progress((i + 1) / len(chunks))

    progress_bar.empty()
    status.write("Synthesizing final structural summary...")

    combined_intermediates = "\n\n--- Chunk Summary Breakdown ---\n\n".join(chunk_summaries)

    final = chain.invoke({
        "text": combined_intermediates,
        "length_guidance": length_guidance[length]
    })
    status.empty()
    return final


def generate_exam_qa(llm: ChatGroq, text: str, num_mcq: int, num_short: int, num_long: int) -> str:
    CHUNK_LIMIT_CHARS = 15000
    chain = QA_PROMPT | llm | StrOutputParser()

    if len(text) <= 25000:
        return chain.invoke({
            "text": text,
            "num_mcq": num_mcq,
            "num_short": num_short,
            "num_long": num_long,
        })

    st.info("⚠️ This is a large document. Generating questions via a multi-stage pipeline...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_LIMIT_CHARS,
        chunk_overlap=1500
    )
    chunks = text_splitter.split_text(text)

    # Distribute requested question counts across chunks, then trim in the final pass.
    per_chunk_mcq = max(1, round(num_mcq / len(chunks))) if num_mcq else 0
    per_chunk_short = max(1, round(num_short / len(chunks))) if num_short else 0
    per_chunk_long = max(1, round(num_long / len(chunks))) if num_long else 0

    chunk_outputs = []
    progress_bar = st.progress(0.0)
    status = st.empty()

    for i, chunk in enumerate(chunks):
        status.write(f"Generating questions from chunk {i + 1} of {len(chunks)}...")
        partial_res = chain.invoke({
            "text": chunk,
            "num_mcq": per_chunk_mcq,
            "num_short": per_chunk_short,
            "num_long": per_chunk_long,
        })
        chunk_outputs.append(partial_res)
        progress_bar.progress((i + 1) / len(chunks))

    progress_bar.empty()
    status.write("Selecting the best questions overall...")

    combined_intermediates = "\n\n--- Chunk Question Set ---\n\n".join(chunk_outputs)

    final = chain.invoke({
        "text": combined_intermediates,
        "num_mcq": num_mcq,
        "num_short": num_short,
        "num_long": num_long,
    })
    status.empty()
    return final


st.title("AI Note Summarizer")
st.caption(
    "Turn notes, documents, or web articles into a summary, key points, and action items - "
    "Powered by LANGCHAIN + GROQ"
)

with st.sidebar:
    st.header("Settings")
    # NOTE: never hardcode API keys in source code. Enter your key below,
    # or set the GROQ_API_KEY environment variable before launching the app.
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Get a free key at https://console.groq.com/keys",
    )
    model = st.selectbox("Model", MODEL, index=0)

    mode = st.radio("Output mode", ["📝 Summary", "🎓 Exam Q&A"], index=0)

    if mode == "📝 Summary":
        length = st.select_slider("Summary length", options=["short", "medium", "detailed"], value="medium")
    else:
        st.caption("Exam Q&A question counts (Sir Syed University style)")
        num_mcq = st.number_input("Multiple Choice Questions", min_value=0, max_value=30, value=10, step=1)
        num_short = st.number_input("Short Questions", min_value=0, max_value=20, value=5, step=1)
        num_long = st.number_input("Long Questions", min_value=0, max_value=10, value=3, step=1)

    st.markdown("Get a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys)")

tab_text, tab_file, tab_url = st.tabs(["✍️ Paste Text", "📄 Upload File", "🔗 From URL"])

pending_source = None
with tab_text:
    pasted = st.text_area(
        "Paste your notes here", height=300, placeholder="Paste meeting notes, lecture notes, an article, etc..."
    )
    text_btn_label = "Summarize Text" if mode == "📝 Summary" else "Generate Exam Q&A"
    if st.button(text_btn_label, key="btn_text", type='primary'):
        if pasted.strip():
            pending_source = ('text', {"text": pasted})
        else:
            st.warning("Please paste some text first.")

with tab_file:
    uploaded = st.file_uploader("Upload a file", type=["pdf", 'txt', 'docx'])
    file_btn_label = "Summarize file" if mode == "📝 Summary" else "Generate Exam Q&A"
    if st.button(file_btn_label, key="btn_file", type="primary"):
        if uploaded:
            pending_source = ("file", {"file": uploaded})
        else:
            st.warning("Please upload a file first.")

with tab_url:
    url = st.text_input("Enter article URL", placeholder="https://example.com")
    url_btn_label = "Summarize URL" if mode == "📝 Summary" else "Generate Exam Q&A"
    if st.button(url_btn_label, key="btn_url", type="primary"):
        if url.strip():
            if url.strip().startswith(("http://", "https://")):
                pending_source = ("url", {"url": url.strip()})
            else:
                st.warning("Please enter a valid URL starting with http:// or https://")
        else:
            st.warning("Please enter a URL first.")

if pending_source:
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    else:
        source_type, kwargs = pending_source
        try:
            with st.spinner("Reading input..."):
                input_text = load_input_text(source_type, **kwargs)
        except Exception as e:
            st.error(f"Couldn't read that input: {e}")
            input_text = ""

        if input_text and input_text.strip():
            spinner_text = "Summarizing with Groq..." if mode == "📝 Summary" else "Generating exam Q&A with Groq..."
            with st.spinner(spinner_text):
                try:
                    llm = get_llm(api_key, model)
                    if mode == "📝 Summary":
                        result = summarize(llm, input_text, length)
                    else:
                        result = generate_exam_qa(llm, input_text, num_mcq, num_short, num_long)
                except Exception as e:
                    st.error(f"Error generating output: {e}")
                    result = None

            if result:
                st.markdown("---")
                st.markdown(result)
                download_label = "⬇️ Download Summary" if mode == "📝 Summary" else "⬇️ Download Exam Q&A"
                file_prefix = "summary" if mode == "📝 Summary" else "exam_qa"
                st.download_button(
                    download_label,
                    data=result,
                    file_name=f"{file_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                )
                with st.expander("View extracted / raw input text"):
                    preview = input_text[:5000]
                    st.text(preview + ("..." if len(input_text) > 5000 else ""))
        elif input_text is not None and not input_text.strip():
            st.error("No text could be extracted from that input.")
