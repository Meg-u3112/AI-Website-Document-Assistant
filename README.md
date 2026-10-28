🤖 AI Website & Document Assistant

An AI-powered application that lets users ask questions about website content and uploaded PDF/TXT documents using Retrieval-Augmented Generation (RAG).

✨ Features

* 🌐 Chat with Websites — Enter a website URL and ask questions about its content.
* 📄 Chat with Documents — Upload PDF or TXT files and ask questions.
* 🧠 Semantic Search — Retrieve relevant content using embeddings and FAISS.
* 🤖 Local AI — Generate answers using Ollama and Llama 3.2.
* 📚 Source References — View relevant content used to generate answers.
* 💻 Interactive UI — Built with Streamlit.

🛠️ Tech Stack

* Python
* Streamlit
* LangChain
* Hugging Face Sentence Transformers
* FAISS
* Ollama
* Llama 3.2
* PyMuPDF
* Beautiful Soup

⚙️ Prerequisites

Install the following before running the project:

* Python
* Git
* Ollama

Download and install Ollama from its official website:
https://ollama.com/

Pull the required model:

ollama pull llama3.2:3b

🚀 Installation

Clone the repository:

git clone https://github.com/Meg-u3112/AI-Website-Document-Assistant.git
cd AI-Website-Document-Assistant

Create a virtual environment:

python3 -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

▶️ Run the Application

python -m streamlit run app/streamlit_app.py

Open the local URL displayed in your terminal.

📖 How to Use

Website Chat

1. Select Website Chat.
2. Enter a website URL.
3. Load the website.
4. Ask questions about its content.

Document Chat

1. Select Document Chat.
2. Upload a PDF or TXT file.
3. Wait for document processing to finish.
4. Ask questions about the uploaded document.
5. Review the answer and relevant sources.

🏗️ Project Structure

AI-Website-Document-Assistant/
├── app/
│   ├── loader.py
│   ├── streamlit_app.py
│   └── vectorstore.py
├── .gitignore
├── README.md
├── main.py
└── requirements.txt

⚠️ Notes

* Ollama must be running locally with the required model available.
* The first embedding-model load may require an internet connection to download model files.
* Website access and text extraction depend on the website’s structure and accessibility.
* Scanned PDFs may require OCR for text extraction.

👨‍💻 Author

Meganaathan G

GitHub: https://github.com/Meg-u3112

⸻

Built with Python, LangChain, Streamlit, FAISS, and Ollama.
