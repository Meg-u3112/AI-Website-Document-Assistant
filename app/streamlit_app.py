import streamlit as st
import pymupdf
from urllib.parse import urlparse
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from loader import load_and_split
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1 - Page configuration
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown(
    """
    <style>
    [data-testid="stSidebar"]{
    display:none;
    }
    [data-testid="collapsedControl"]{
        display:none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 2 - Create embeddings and vector store
@st.cache_resource(ttl=3600)
def create_vectorstore(url):
    chunks = load_and_split(url)

    if not chunks:
        raise ValueError("No content could be extracted from this website.")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.from_documents(chunks, embedding_model)


# 3 - Load the language model
@st.cache_resource
def load_rag():
    return ChatOllama(
        model="llama3.2:3b",
        temperature=0
    )


llm = load_rag()


# 4 - Helper: load a website
def load_website(url):
    url = url.strip()
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        st.warning("Please enter a valid HTTP or HTTPS URL.")
        return

    try:
        with st.spinner("Reading website and building index..."):
            st.session_state.vectorstore = create_vectorstore(url)
            st.session_state.loaded_url = url
            st.session_state.chat_history = []

        st.success("Website loaded successfully!")

    except Exception as e:
        st.error(f"Could not load website: {e}")


# 5 - Session state
if "page" not in st.session_state:
    st.session_state.page = "home"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "loaded_url" not in st.session_state:
    st.session_state.loaded_url = None

if "document_vectorstore" not in st.session_state:
    st.session_state.document_vectorstore = None

if "document_text" not in st.session_state:
    st.session_state.document_text = ""

if "document_name" not in st.session_state:
    st.session_state.document_name = ""




# 7 - Main area

# HOME PAGE
if st.session_state.page == "home":

    st.title("🤖 AI Assistant")
    st.write("Choose a source to start asking questions.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🌐 Website Chat")
        st.write("Ask questions about information available on a website.")

        if st.button("Open Website Chat →", use_container_width=True):
            st.session_state.page = "website"
            st.rerun()

    with col2:
        st.subheader("📄 Document Chat")
        st.write("Upload a PDF or TXT file and ask questions about its content.")

        if st.button("Open Document Chat →", use_container_width=True):
            st.session_state.page = "document"
            st.rerun()


# DOCUMENT CHAT PAGE
elif st.session_state.page == "document":

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.title("📄 Chat with Documents")
    st.write("Upload a PDF or TXT file and ask questions about its content.")

    uploaded_file = st.file_uploader(
        "Upload your document",
        type=["pdf", "txt"],
        key="document_uploader"
    )

    if uploaded_file is not None:

        try:
            if uploaded_file.name.lower().endswith(".txt"):
                text = uploaded_file.getvalue().decode("utf-8")

            elif uploaded_file.name.lower().endswith(".pdf"):
                pdf_doc = pymupdf.open(
                    stream=uploaded_file.getvalue(),
                    filetype="pdf"
                )

                text = "\n\n".join(
                    page.get_text() for page in pdf_doc
                )

                pdf_doc.close()

            else:
                text = ""

            if not text.strip():
                st.warning("No readable text found in this document.")

            else:
                # Process the document only when a new file is uploaded
                if (
                    st.session_state.document_name != uploaded_file.name
                    or st.session_state.document_vectorstore is None
                ):
                    with st.spinner("Processing document..."):

                        document = Document(
                            page_content=text,
                            metadata={"source": uploaded_file.name}
                        )

                        text_splitter = RecursiveCharacterTextSplitter(
                            chunk_size=200,
                            chunk_overlap=50
                        )

                        chunks = text_splitter.split_documents([document])

                        embedding_model = HuggingFaceEmbeddings(
                            model_name="sentence-transformers/all-MiniLM-L6-v2"
                        )

                        st.session_state.document_vectorstore = (
                            FAISS.from_documents(chunks, embedding_model)
                        )

                        st.session_state.document_text = text
                        st.session_state.document_name = uploaded_file.name
                        st.session_state.document_chat_history = []

                    st.success("Document indexed successfully!")

                with st.expander("📄 Document Details"):
                    st.write("**File name:**", uploaded_file.name)
                    st.write("**File size:**", f"{uploaded_file.size:,} bytes")
                    st.write("**Total chunks:**", len(chunks) if "chunks" in locals() else "Indexed")

                st.divider()
                st.subheader("💬 Ask Your Document")

                query = st.text_input(
                    "Enter your question",
                    placeholder="e.g., What is the discount policy?",
                    key="document_question"
                )

                if st.button("Ask AI", type="primary"):

                    if query.strip():

                        results = st.session_state.document_vectorstore.similarity_search(
                            query,
                            k=3
                        )

                        context = "\n\n".join(
                            result.page_content for result in results
                        )

                        prompt = f"""
You are a document question-answering assistant.

Answer using only the provided document context.
If the answer is not available in the context, say you could not find it in the uploaded document.
Do not make up information.

Document Context:
{context}

Question:
{query}

Answer:
"""

                        with st.spinner("Finding your answer..."):
                            answer = llm.invoke(prompt).content

                        st.subheader("🤖 AI Answer")
                        st.write(answer)

                        with st.expander("📚 View relevant sources"):
                            for i, result in enumerate(results, start=1):
                                st.markdown(f"**Source {i}**")
                                st.write(result.page_content)
                                st.caption(
                                    f"File: {result.metadata.get('source', 'Unknown')}"
                                )

                    else:
                        st.warning("Please enter a question.")
        except Exception as e:
            st.error(f"Error processing document: {e}")

# WEBSITE CHAT PAGE
elif st.session_state.page == "website":

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.title("🌐 Website Assistant")
    st.write("Enter a website URL and ask questions about its content.")

    # Website URL input in the main page
    with st.form("load_form"):
        web_url = st.text_input(
            "Website URL",
            value=st.session_state.loaded_url or "",
            placeholder="https://en.wikipedia.org/wiki/Car"
        )

        load_clicked = st.form_submit_button(
            "🌐 Load Website",
            type="primary",
            use_container_width=True
        )

    if load_clicked:
        load_website(web_url)

    # Example website
    if st.button("📖 Try Example: Wikipedia Car"):
        load_website("https://en.wikipedia.org/wiki/Car")
        st.rerun()

    # Show loaded website
    if st.session_state.vectorstore is not None:
        domain = urlparse(st.session_state.loaded_url).netloc
        st.success(f"Connected to: {domain}")

        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])

        with st.chat_message("assistant"):
            st.write(chat["answer"])

    # Chat input at the bottom
    if qt := st.chat_input(
        "Ask about the website...",
        disabled=st.session_state.vectorstore is None
    ):

        with st.chat_message("user"):
            st.write(qt)

        try:
            history = ""

            for chat in st.session_state.chat_history[-3:]:
                history += (
                    f"Previous Question: {chat['question']}\n"
                    f"Previous Answer: {chat['answer']}\n\n"
                )

            search_query = qt

            if st.session_state.chat_history:
                rewrite_prompt = f"""
You are a question-rewriting assistant.
Rewrite the user's latest question as a standalone question
using the conversation history.

Conversation History:
{history}

Latest Question:
{qt}

Return only the rewritten question. Do not answer it.
"""

                search_query = llm.invoke(
                    rewrite_prompt
                ).content.strip()

            retriever = st.session_state.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 20}
            )

            results = retriever.invoke(search_query)

            context = "\n\n".join(
                doc.page_content for doc in results
            )

            prompt = f"""
You are a helpful website question-answering assistant.

Instructions:
- Answer using only the website context.
- Use conversation history to understand follow-up questions.
- If the answer is unavailable in the website context,
  say you couldn't find it in the loaded website.
- Do not make up information.

Conversation History:
{history}

Website Context:
{context}

Current Question:
{qt}

Answer:
"""

            with st.chat_message("assistant"):
                answer = st.write_stream(
                    chunk.content for chunk in llm.stream(prompt)
                )

            st.session_state.chat_history.append({
                "question": qt,
                "answer": answer
            })

            with st.expander("📚 Sources"):
                for i, doc in enumerate(results, start=1):
                    st.markdown(
                        f"**Source {i}:** "
                        f"{doc.metadata.get('source', 'Unknown')}"
                    )
                    st.write(doc.page_content)

        except Exception as e:
            st.error(f"Error generating answer: {e}")
