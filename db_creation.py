import os
import langchain
import openai

from langchain_community.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
#from langchain.embeddings import OpenAIEmbeddings
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms import OpenAI


os.environ["OPENAI_API_KEY"] = ''

embedding = OpenAIEmbeddings()

def load_docs(data_dir):
  loader = DirectoryLoader(data_dir)
  documents = loader.load()
  return documents

data_dir = "/home/chaitra/QA/Data"
#data_dir ="/Users/chaitrakaustubh/Documents/KG/QA_LangChain/Data"
documents = load_docs(data_dir)
print(len(documents))

def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))

persist_directory = 'db'

embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents( documents=docs,
                                 embedding=embedding,
                                 persist_directory=persist_directory)

#vectordb.persist()
