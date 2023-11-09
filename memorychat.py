import re
import time
from io import BytesIO
from typing import List
import openai
import streamlit as st
import sqlite3  # Add SQLite for database storage
from langchain.chains import LLMChain
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from pypdf import PdfReader
import os


api = st.secrets['OPENAI_API_KEY']


# Initialize SQLite database for chat history
conn = sqlite3.connect('chat_history.db')
c = conn.cursor()


# Create a table for chat history if it doesn't exist
c.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        query TEXT,
        response TEXT
    )
''')
conn.commit()

# Define a function to insert chat messages into the database
def insert_chat(query, response):
    c.execute('INSERT INTO chat_history (query, response) VALUES (?, ?)', (query, response))
    conn.commit()

# Define a function to fetch chat history from the database
def fetch_chat_history():
    c.execute('SELECT query, response FROM chat_history')
    history = c.fetchall()
    return history



# Define a function to parse a PDF file and extract its text content
@st.cache_data
def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        output.append(text)
    return output





# Define a function to convert text content to a list of documents
@st.cache_data
def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources as metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks








# Define a function for the embeddings
@st.cache_data
def create_embeddings():
    embeddings = OpenAIEmbeddings(openai_api_key=api)
    # Indexing
    # Save in a Vector DB
    with st.spinner("It's indexing..."):
        index = FAISS.from_documents(pages, embeddings)
    st.success("Embeddings done.", icon="✔")
    return index




# Define a function to clear the conversation history
def clear_history():
    st.session_state.memory.clear()





# Set up the Streamlit app
st.title("MultiTurn ChatBot-Ask any questions from your PDF by uploading it")




# Allow the user to upload a PDF file
uploaded_file = st.file_uploader("**Upload Your PDF File**", type=["pdf"])



if uploaded_file:
    name_of_file = uploaded_file.name
    doc = parse_pdf(uploaded_file)
    pages = text_to_docs(doc)

    if pages:
        # Allow the user to select a page and view its content
        with st.expander("Show Page Content", expanded=False):
            page_sel = st.number_input(label="Select Page", min_value=1, max_value=len(pages), step=1)
            st.write(pages[page_sel - 1])



        if api:
            # Test the embeddings and save the index in a vector database
            index = create_embeddings()
            # Set up the question-answering system
            qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=api), chain_type="stuff", retriever=index.as_retriever())



            # Set up the conversational agent
            tools = [
            Tool(name="Document Q&A Tool", func=qa.run,
                       description="This tool allows you to ask questions about the document you've uploaded. You can ask about any topic or content within the document.",
                 )]
            prefix = """Engage in a conversation with the AI, answering questions about the uploaded document. You have access to a single tool:"""
            suffix = """Begin the conversation!"

            {chat_history}
            Question: {input}
            {agent_scratchpad}"""



            prompt = ZeroShotAgent.create_prompt(tools,prefix=prefix, suffix=suffix,
                                                  input_variables=["input", "chat_history", "agent_scratchpad"])


            if "memory" not in st.session_state:
                st.session_state.memory = ConversationBufferMemory(memory_key="chat_history")


            llm_chain = LLMChain(llm=OpenAI(temperature=0, openai_api_key=api, model_name="gpt-3.5-turbo"), prompt=prompt)


            agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
            agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=st.session_state.memory)
            


            # Allow the user to enter a query and generate a response
            query = st.text_input("Start a Conversation with the Bot!", placeholder="Ask the bot anything from {}".format(name_of_file))


            if query:
                with st.spinner("Generating Answer to your Query: `{}`".format(query)):
                    res = agent_chain.run(query)
                    st.info(res, icon="📝")
            
                    # Store the conversation in the database
                    insert_chat(query, res)


# Display conversation history in a sidebar
st.sidebar.title("Conversation History")
history = fetch_chat_history()

for query, response in history:
    st.sidebar.text(f"User: {query}")
    st.sidebar.text(f"Bot: {response}")
    st.sidebar.text("------")

# Add a "New Chat" button to clear the conversation history
if st.button("New Chat"):
    clear_history()

