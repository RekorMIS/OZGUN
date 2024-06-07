from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# define what documents to load
loader = DirectoryLoader(path='C:\\Users\\osungar\\Desktop\\projects\\chatbot\\pythonFiles\\demo\\data\\pdfs\\', glob="*.pdf", loader_cls=PyPDFLoader)

# interpret information in the documents
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                          chunk_overlap=50)
texts = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'})
# create the vector store database
db = FAISS.from_documents(texts, embeddings)
# save the vector store
db.save_local("C:\\Users\\osungar\\Desktop\\projects\\chatbot\\pythonFiles\\demo\\data\\faiss\\")