#!/usr/bin/env python
# coding: utf-8

# In[16]:


from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


# In[10]:


import arxiv
# papers about LLMs
paper_ids = ['2308.10620', '2307.06435', '2303.18223', '2307.10700', '2310.11207', '2305.11828']
for paper_id in paper_ids:
    search = arxiv.Search(id_list=[paper_id])
    paper = next(search.results())
    print(paper.title)
    
    paper.download_pdf('C:\\Users\\osungar\\Desktop\\projects\\chatbot\\data\\')


# In[18]:


from langchain.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS


# In[19]:


# define what documents to load
loader = DirectoryLoader(path='C:\\Users\\osungar\\Desktop\\projects\\chatbot\\data\\', glob="*.pdf", loader_cls=PyPDFLoader)


# In[20]:


# interpret information in the documents
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                          chunk_overlap=50)


# In[21]:


texts = splitter.split_documents(documents)


# In[22]:


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})


# In[23]:


# create the vector store database
db = FAISS.from_documents(texts, embeddings)


# In[27]:


# save the vector store
db.save_local("C:\\Users\\osungar\\Desktop\\projects\\chatbot\\data\\faiss\\")


# In[28]:


#transformer modeli C dilinde yapılmış
from langchain.llms import CTransformers
def load_llm(): 
  #load the lm
  llm = CTransformers(model='C:\\Users\\osungar\\Desktop\\projects\\chatbot\\data\\models\\llama-2–7b-chat.ggmlv3.q2_K.bin',
  # model available here: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
  model_type='llama',
  config={'max_new_tokens': 256, 'temperature': 0})
  return llm   


# In[29]:


def load_vector_store(): 
  # load the vector store
  embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'})
  db = FAISS.load_local('faiss', embeddings)
  return db


# In[30]:


#prompt eklenmesi
from langchain import PromptTemplate
def create_prompt_template():
  # prepare the template that provides instructions to the chatbot
  template = """Use the provided context to answer the user’s question.
                If you don’t know the answer, respond with "I do not know".
                Context: {context}
                History: {history}
                Question: {question}
                Answer: """
  prompt = PromptTemplate(
                template=template,
                input_variables=['context', 'question'])
  return prompt


# In[31]:


#soru cevap zinciri
from langchain.chains import RetrievalQA
def create_qa_chain():
  """create the qa chain"""
  # load the llm, vector store, and the prompt
  llm = load_llm()
  db = load_vector_store()
  prompt = create_prompt_template()
  # create the qa_chain
  retriever = db.as_retriever(search_kwargs={'k': 2})
  qa_chain = RetrievalQA.from_chain_type(llm=llm,
                            chain_type='stuff',
                            retriever=retriever,
                            return_source_documents=True,
                            chain_type_kwargs={'prompt': prompt})
  return qa_chain


# In[33]:


# cevap döndürecek fonksiyon
def generate_response(query, qa_chain):
  # use the qa_chain to answer the given query
  return qa_chain({'query':query})['result']


# In[ ]:




