from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from loader import load_and_split

# website
url = "https://www.bbc.com/future/article/20260902-how-to-watch-the-cameras-that-watch-you"

# load and split the website
chunks = load_and_split(url)

# embedding_model
embedding_model = HuggingFaceEmbeddings(
  model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

# create FAISS vector store
vectorstore = FAISS.from_documents(
  chunks,
  embedding_model
)
print("No of chunks:",len(chunks))
print("FAISS vector store")