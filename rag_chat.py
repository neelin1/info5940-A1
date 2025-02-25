import streamlit as st
from openai import OpenAI
from os import environ
import PyPDF2
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Threshold for including
# MAX_INLINE_LENGTH = 2000
NUM_CHUNKS=3
MAX_CHARACTERS_SUMMARY = 6000
client = OpenAI(api_key=environ["OPENAI_API_KEY"])

st.title("Chat with Files")
uploaded_files = st.file_uploader(
    "Upload articles",
    type=("txt", "md", "pdf"),
    accept_multiple_files=True
)

question = st.chat_input(
    "Ask a question about the uploaded documents",
    disabled=not uploaded_files,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask a question about the uploaded documents"}]
if "files_data" not in st.session_state:
    st.session_state["files_data"] = []
# if "vectorstore" not in st.session_state:
#     st.session_state["vectorstore"] = None
# if "full_text" not in st.session_state:
#     st.session_state["full_text"] = ""

# # Fetch the available models
# models = client.models.list()

# # Print the model names
# for model in models:
#     print(model.id)


def get_file_text(file):
    if file.name.lower().endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        
        reader = PyPDF2.PdfReader(tmp_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        return file.read().decode("utf-8")

def summarize_text(text, openai_api_key):
    truncated_text = text[:MAX_CHARACTERS_SUMMARY]  

    system = "You are an expert summarizer. Summarize the following text in one long paragraph. Please omit needless words. Vigorous writing is concise. A sentence should contain no unnecessary words, a paragraph no unnecessary sentences, for the same reason that a drawing should have no unnecessary lines and a machine no unnecessary parts. This requires not that the writer make all their sentence short, or that they avoid all detail and treat their subject only in outline, but that they make every word tell."

    if (len(text) > MAX_CHARACTERS_SUMMARY):
        system += f"\nYou will see at most {MAX_CHARACTERS_SUMMARY} characters, so try to extrapolate outward about missing content (there might be a table of contents or some other useful feature)."

    response = client.chat.completions.create(
        model="openai.gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": truncated_text}
        ]
    )

    return response.choices[0].message.content

def build_vectorstore_for_text(text, openai_api_key):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    if not chunks:
        return None
    
    embeddings = OpenAIEmbeddings(
        model="openai.text-embedding-3-large",
        api_key=environ["OPENAI_API_KEY"]
    )
    return FAISS.from_texts(chunks, embeddings)
    

def process_files(files):
    st.session_state["files_data"].clear()

    for f in files:
        raw_text = get_file_text(f)
        summary = summarize_text(raw_text, environ["OPENAI_API_KEY"])
        vectorstore = build_vectorstore_for_text(raw_text, environ["OPENAI_API_KEY"])

        st.session_state["files_data"].append({
            "filename": f.name,
            "summary": summary,
            "vectorstore": vectorstore
        })

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if uploaded_files:
    process_files(uploaded_files)

if question and uploaded_files:
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    system_prompt = "The user has uploaded the following files.\n"
    relevant_chunks_text = ""

    for file_info in st.session_state["files_data"]:
        filename = file_info["filename"]
        summary = file_info["summary"]
        vectorstore = file_info["vectorstore"]

        system_prompt += f"\n# File Name: {filename}\nSummary: {summary}\n"

        if vectorstore:
            docs = vectorstore.similarity_search(question, k=NUM_CHUNKS)
            rag_text = "\n---\n\n".join([d.page_content for d in docs])
            relevant_chunks_text += f"\n## RAG for {filename}\n{rag_text}\n"

    system_prompt += f"\nHere are the relevant chunks from the documents:\n{relevant_chunks_text}\n"

    print(system_prompt)

    client = OpenAI(api_key=environ["OPENAI_API_KEY"])

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="openai.gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                *st.session_state.messages
            ],
            stream=True
        )
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})
