# Instructions to Run
Open `docker-compose.yml` and replace <API-KEY> with your OpenAI API key.

Run and build the dokcer container with:
`docker-compose build devcontainer`
`docker-compose up -d devcontainer`

Make sure you have the VSCode extension `Dev Containers` and `cmd+shift+p` and run the command `Dev Containers: Reopen in Container`

To make sure this is working, type `echo $OPENAI_API_KEY` and make 

Run `streamlit run rag_chat.py --server.port 8501 --server.address 0.0.0.0` and open `http://0.0.0.0:8501/` in a broswer to see the application.

# Overview of Implementation
I implement RAG with a summarizing feature in `rag_chat.py`. Every file that gets uploaded is tokenized with `openai.text-embedding-3-large`. Its first 6000 characters then get summarized into a single paragraph for overall context on the file. Then when the user asks a question, the system prompt sees each file's name, this one-paragraph contexual summary, and an n=3 RAG similarity search. I think this one-paragraph summary provides necessary context on the RAG retreival.

For the summary, I use the followign system prompt. I frequently use this prompt. I find that LLMs often are overly verbose, and can put more information into the same size paragraph if prompted with something like this:
> "You are an expert summarizer. Summarize the following text in one long paragraph. Please omit needless words. Vigorous writing is concise. A sentence should contain no unnecessary words, a paragraph no unnecessary sentences, for the same reason that a drawing should have no unnecessary lines and a machine no unnecessary parts. This requires not that the writer make all their sentence short, or that they avoid all detail and treat their subject only in outline, but that they make every word tell."
