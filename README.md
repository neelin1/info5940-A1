# Instructions to Run
Open `docker-compose.yml` and replace <API-KEY> with your OpenAI API key.

Run and build the dokcer container with:
`docker-compose build devcontainer`
`docker-compose up -d devcontainer`

Make sure you have the VSCode extension `Dev Containers` and `cmd+shift+p` and run the command `Dev Containers: Reopen in Container`

To make sure this is working, type `echo $OPENAI_API_KEY` and make 

Run `streamlit run rag_chat.py --server.port 8501 --server.address 0.0.0.0` and open `http://0.0.0.0:8501/` in a broswer to see the application.
