import requests
from requests.exceptions import Timeout,HTTPError,RequestException
from bs4 import BeautifulSoup

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split(url):
    """
    Load a website, extract its readable content,
    and split the content into smaller chunks.
    """

    if not url:
        raise ValueError("URL cannot be empty.")

    # 1. Request the website
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        )
    }

    #  1. request the webs
    try:
        response = requests.get(
            url,
            headers = headers,
            timeout=20
        )
        response.raise_for_status()
    except Timeout:
        raise ValueError(
            "The website took too long to respond. Please try again."
        )
    except HTTPError as e:
        status_code = e.response.status_code
        if status_code == 403:
            raise ValueError(
                "Access denied by the website (403). "
                "The website may block automated requests."
            )
        elif status_code == 404:
            raise ValueError(
                "Page not found(404). please check URL."
            )
        else:
            raise ValueError(
                f"website returned as HTTP error: {status_code}."
            )
    except RequestException as e:
        raise ValueError(
            f"Could not connecet to the website: {e}"
        )

    # 2. Parse HTML
    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # 3. Remove unnecessary elements
    for element in soup.find_all(
        ["script", "style", "nav", "footer", "button", "header","aside","from","noscript"]
    ):
        element.decompose()

    # 4. Extract the main content
    article = soup.find("article")

    if article is None:
        # Wikipedia and other websites may use <main>
        article = soup.find("main")

    if article is None:
        # Final fallback: use the page body
        article = soup.find("body")

    if article is None:
        raise ValueError(
            "Could not find readable content on this website."
        )

    text = article.get_text(
        separator=" ",
        strip=True
    )

    if not text:
        raise ValueError(
            "The website returned no readable text."
        )

    # 5. Create a LangChain document
    document = Document(
        page_content=text,
        metadata={
            "source": url
        }
    )

    # 6. Split the content into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        [document]
    )

    print(f"Website loaded: {url}")
    print(f"Total chunks: {len(chunks)}")

    return chunks