# Instructions to Run
Open `docker-compose.yml` and replace <API-KEY> with your OpenAI API key.

Run and build the dokcer container with:
`docker-compose build devcontainer`
`docker-compose up -d devcontainer`

Make sure you have the VSCode extension `Dev Containers` and `cmd+shift+p` and run the command `Dev Containers: Reopen in Container`

To make sure this is working, type `echo $OPENAI_API_KEY` and make 

Run `streamlit run rag_chat.py --server.port 8501 --server.address 0.0.0.0` and open `http://0.0.0.0:8501/` in a broswer to see the application.

# Overview of Implementation
I implement RAG with a summarizing feature. Every file that gets uploaded is tokenized with `openai.text-embedding-3-large`. Its first 6000 characters then get summarized into a single paragraph for overall context on the file. Then when the user asks a question, the system prompt sees each file's name, this one-paragraph contexual summary, and an n=3 RAG similarity search. I think this one-paragraph summary provides necessary context on the RAG retreival.