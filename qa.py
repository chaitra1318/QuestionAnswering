import os
import langchain
import openai
import warnings
import sys

from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

import shutup
shutup.please()

#warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain.chat_models")


os.environ["OPENAI_API_KEY"] = ''


#embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory="./db", embedding_function=OpenAIEmbeddings())

retriever= vectordb.as_retriever(searchtype="mmr",k=3)

model = "gpt-4o"
#llm = OpenAI(model=model)
llm = ChatOpenAI(model="gpt-4o")

# Build prompt
template = """
Answer the query in not more than 15 words. Provide 3 follow up questions from the retrieved text.{context}
Question: {question}
Helpful Answer: """
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)
question =sys.argv[1]
#print(question)
result = qa_chain({"query": question})
print(result["result"])
