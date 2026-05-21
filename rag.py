from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
apikey = os.getenv("GROQ_API_KEY")
    

def get_pdf(name_of_pdf):
    loader = PyPDFLoader(name_of_pdf)
    pdf = loader.load()
    return pdf

def split_document(pdf):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pdf)
    return chunks

def create_retriever(chunks):
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return retriever


def initialize_llm():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=apikey
    )
    return llm

def create_prompt():
    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided context.
        If the context is insufficient, just say you don't know.

        {context}
        Question: {question}
        """,
        input_variables = ['context', 'question']
    )
    return prompt

def format_docs(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


def build_chain(retriever,prompt,llm):
    parallel_chain = RunnableParallel({
        'context': retriever | RunnableLambda(format_docs),
        'question': RunnablePassthrough()
    })

    parser = StrOutputParser()

    main_chain = parallel_chain | prompt | llm | parser
    return main_chain


def load_pdf(NameOfPdf):
    pdf = get_pdf(NameOfPdf)

    chunks = split_document(pdf)

    retriver = create_retriever(chunks)

    llm = initialize_llm()

    prompt = create_prompt()

    chain = build_chain(retriver,prompt,llm)

    return chain

def answer(qustion,chain):
    answer = chain.invoke(qustion)
    return answer