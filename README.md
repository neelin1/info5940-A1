# Instructions to Run

Open `docker-compose.yml` and replace <API-KEY> with your OpenAI API key.

Run and build the dokcer container with:
`docker-compose build devcontainer`
`docker-compose up -d devcontainer`

Make sure you have the VSCode extension `Dev Containers` and `cmd+shift+p` and run the command `Dev Containers: Reopen in Container`

To make sure this is working, type `echo $OPENAI_API_KEY` and you should see the API key

Run `streamlit run rag_chat.py --server.port 8501 --server.address 0.0.0.0` and open `http://0.0.0.0:8501/` in a broswer to see the application. Make sure that port 8501 is free.

# Requirements
I used the exact same requirements as found in the original `pyproject.toml` of lecture-05 with the exception of adding:
```
pypdf2 = "^3.0.1"
faiss-cpu = "^1.10.0"
```

The full list of dependencies is as follows:
```
python = "^3.11"
aioboto3 = "^12"
fsspec = "*"
openai = "^1.14"
pandas = "^2"
poetry = "^1.7"
pydantic = "^2"
s3fs = "*"
streamlit = "^1.0"
tiktoken = "^0.7.0"
langchain-community = "^0.2.15"
langchain = "^0.2.15"
langchain_core = "^0.2.15"
langchain-openai = "^0.1.23"
pydub = "^0.25.1"
scipy = "^1.14.1"
langchain-chroma = "^0.1.3"
pypdf = "^4.3.1"
Markdown = "^3.7"
pypdf2 = "^3.0.1"
faiss-cpu = "^1.10.0"
```

# Overview of Implementation

I implement RAG with a summarizing feature in `rag_chat.py`. Every file that gets uploaded is tokenized with `openai.text-embedding-3-large`. Its first 6000 characters then get summarized into a single paragraph for overall context on the file. Then when the user asks a question, the system prompt sees each file's name, this one-paragraph contexual summary, and an n=3 RAG similarity search. I think this one-paragraph summary provides necessary context on the RAG retreival.

For the summary, I use the followign system prompt. I frequently use this prompt. I find that LLMs often are overly verbose, and can put more information into the same size paragraph if prompted with something like this:

> "You are an expert summarizer. Summarize the following text in one long paragraph. Please omit needless words. Vigorous writing is concise. A sentence should contain no unnecessary words, a paragraph no unnecessary sentences, for the same reason that a drawing should have no unnecessary lines and a machine no unnecessary parts. This requires not that the writer make all their sentence short, or that they avoid all detail and treat their subject only in outline, but that they make every word tell."
