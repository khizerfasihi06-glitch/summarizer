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
MODEL = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

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
    
    for i, chunk in enumerate(chunks):
        st.write(f"Processing chunk {i+1} of {len(chunks)}...")
        partial_res = chain.invoke({
            "text": chunk, 
            "length_guidance": "Keep this component snippet dense and concise."
        })
        chunk_summaries.append(partial_res)
        progress_bar.progress((i + 1) / len(chunks))
        
    progress_bar.empty()
    
    combined_intermediates = "\n\n--- Chunk Summary Breakdown ---\n\n".join(chunk_summaries)
    st.write("Synthesizing final structural template...")
    
    return chain.invoke({
        "text": combined_intermediates, 
        "length_guidance": length_guidance[length]
    })


st.title("AI Note Summarizer")
st.caption(
    "Turn notes, documents, or web articles into a summary, key points, and action items - "
    "Powered by LANGCHAIN + GROQ"
)

with st.sidebar:
    st.header("Settings")
    os.environ["GROQ_API_KEY"] = "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"
    api_key = st.text_input("Groq_API_KEY", type="password")
    model = st.selectbox("Model", MODEL, index=0)
    length = st.select_slider("Summary length", options=["short", "medium", "detailed"], value="medium")

    st.markdown("Get a free Groq API key at [://groq.com](https://://groq.com)")

tab_text, tab_file, tab_url = st.tabs(["✍️ Paste Text", "📄 Upload File", "🔗 From URL"])

pending_source = None
with tab_text:
    pasted = st.text_area(
        "Paste your notes here", height=300, placeholder="Paste meeting notes, lecture notes, an article, etc..."
    )
    if st.button("Summarize Text", key="btn_text", type='primary'):
        if pasted.strip():
            pending_source = ('text', {"text": pasted})
        else:
            st.warning("Please paste some text first.")

with tab_file:
    uploaded = st.file_uploader("Upload a file", type=["pdf", 'txt', 'docx'])
    if st.button("Summarize file", key="btn_file", type="primary"):
        if uploaded:
            pending_source = ("file", {"file": uploaded})
        else:
            st.warning("Please upload a file first.")

with tab_url:
    url = st.text_input("Enter article URL", placeholder="https://example.com")
    if st.button("Summarize URL", key="btn_url", type="primary"):
        if url.strip():
            pending_source = ("url", {"url": url.strip()})
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
            with st.spinner("Summarizing with Groq..."):
                try:
                    llm = get_llm(api_key, model)
                    result = summarize(llm, input_text, length)
                except Exception as e:
                    st.error(f"Error generating summary: {e}")
                    result = None

            if result:
                st.markdown("---")
                st.markdown(result)
                st.download_button(
                    "⬇️ Download Summary",
                    data=result,
                    file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                )
                with st.expander("View extracted / raw input text"):
                    preview = input_text[:5000]
                    st.text(preview + ("..." if len(input_text) > 5000 else ""))
        elif input_text is not None and not input_text.strip():
            st.error("No text could be extracted from that input.")
