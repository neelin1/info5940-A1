import streamlit as st
from openai import OpenAI
from os import environ
import PyPDF2
import tempfile

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Threshold for including
MAX_INLINE_LENGTH = 2000
MAX_CHARACTERS_SUMMARY = 6000
client = OpenAI(api_key=environ["OPENAI_API_KEY"])

st.title("Chat with Files")
uploaded_file = st.file_uploader("Upload an article", type=("txt", "md", "pdf"), accept_multiple_files=True)

question = st.chat_input(
    "Ask a question about the uploaded documents",
    disabled=not uploaded_file,
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Ask a question about the uploaded documents"}]
if "files_data" not in st.session_state:
    st.session_state["files_data"] = []
# if "vectorstore" not in st.session_state:
#     st.session_state["vectorstore"] = None
# if "full_text" not in st.session_state:
#     st.session_state["full_text"] = ""
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

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

    if (text.length > MAX_CHARACTERS_SUMMARY):
        system += f"You will see at most {MAX_CHARACTERS_SUMMARY} characters, so try to extrapolate outward about missing content (there might be a table of contents or some other useful feature)."

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert summarizer. Summarize the following text in one long paragraph. Please omit needless words. Vigorous writing is concise. A sentence should contain no unnecessary words, a paragraph no unnecessary sentences, for the same reason that a drawing should have no unnecessary lines and a machine no unnecessary parts. This requires not that the writer make all their sentence short, or that they avoid all detail and treat their subject only in outline, but that they make every word tell. You will see at most 6000 characters, so try to extrapolate outward about missing content (there might be a table of contents or some other useful feature)."},
            {"role": "user", "content": truncated_text}
        ]
    )

    return response.choices[0].message.content
    
def process_file(text):
    # chunk if greater than 2000 characters, otherwise keep as full text
    if len(text) <= MAX_INLINE_LENGTH:
        print("PROCESSING: Text")
        st.session_state["vectorstore"] = None
        st.session_state["full_text"] = text
    else:
        print("PROCESSING: Chunks")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(text)
        
        embeddings = OpenAIEmbeddings(
            model="openai.text-embedding-3-large",
            api_key=environ["OPENAI_API_KEY"]
        )

        st.session_state["vectorstore"] = FAISS.from_texts(chunks, embeddings)
        st.session_state["full_text"] = ""

if uploaded_file:
    text = get_file_text(uploaded_file)
    process_file(text) 

if question and uploaded_file:  
    # client = OpenAI(api_key=environ['OPENAI_API_KEY'])

    # Append the user's question to the messages
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    if st.session_state["vectorstore"]:
        docs = st.session_state["vectorstore"].similarity_search(question, k=2)
        retrieved_text = "\n\n".join([d.page_content for d in docs])
        system_content = f"Relevant chunks:\n{retrieved_text}\n\nAnswer the user query using the chunks above."
    else:
        system_content = f"Here's the file content:\n{st.session_state['full_text']}"
    print ("SYSTEM CONTENT: ", system_content)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="openai.gpt-4o",  # Change this to a valid model name
            messages=[
                {"role": "system", "content": system_content},
                *st.session_state.messages
            ],
            stream=True
        )
        response = st.write_stream(stream)

    # Append the assistant's response to the messages
    st.session_state.messages.append({"role": "assistant", "content": response})


