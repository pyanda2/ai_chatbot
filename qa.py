import os
from typing import List
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import LlamaCpp
from langchain.chains import (
    ConversationalRetrievalChain,
)
from io import BytesIO
import PyPDF2
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory

import chainlit as cl

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
llama_model_path = "models/vicuna-7b-v1.5.Q4_K_M.gguf"

def extract_text_from_pdf(file):
    texts = []
    # with open(pdf_path, 'rb') as pdf_file:
    # with file as pdf_file:
    pdf_reader = PyPDF2.PdfReader(file)

    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        texts.append(page.extractText())

    return texts

@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]
    msg = cl.Message(
        content=f"Processing `{file.name}`...", disable_human_feedback=True
    )
    await msg.send()

    pdf_file = PyPDF2.PdfReader(BytesIO(file.content))

    # Get the total number of pages in the PDF file.
    num_pages = len(pdf_file.pages)

    # Create a list to store the decoded text from each page.
    decoded_text = []

    # Decode each page of the PDF file and add the decoded text to the list.
    for i in range(num_pages):
        page = pdf_file.pages[i]
        decoded_text.append(page.extract_text())

    chunks = []
    for text in decoded_text:
        chunks.extend(text_splitter.split_text(text))
    # Split the text into chunks
    # texts = text_splitter.split_text(text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(chunks))]


    # Create a Chroma vector store
    model_name = "BAAI/bge-small-en"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    embeddings = hf
    docsearch = await cl.make_async(Chroma.from_texts)(
        chunks, embeddings, metadatas=metadatas
    )

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])



    n_gpu_layers = 1
    n_batch = 512 
    
    llm = LlamaCpp(
        model_path=llama_model_path,
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        n_ctx=4096,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        callback_manager=callback_manager,
        verbose=True, # Verbose is required to pass to the callback manager
    )
    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
