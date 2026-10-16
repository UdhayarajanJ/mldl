# To implement text processing using Beautiful Soup.

from bs4 import BeautifulSoup


def lab_15():
    """
    Lab 15: Web Scraping and HTML Parsing with BeautifulSoup

    This lab demonstrates how to perform web scraping and HTML parsing using the BeautifulSoup library in Python.
    It covers loading HTML content, extracting specific elements, stripping HTML tags, and retrieving hyperlinks.

    Steps:
    1. Load a sample HTML document (simulating web page content).
    2. Parse the HTML content using BeautifulSoup.
    3. Extract the title of the document.
    4. Extract specific tag content (e.g., headings, paragraphs).
    5. Strip HTML tags to retrieve clean text.
    6. Extract hyperlinks (href attributes) from anchor tags.
    """
    # Sample HTML document (simulating web page content)
    html_content = """
    <html>
    <head>
        <title>MCA Lab - Web Scraping</title>
    </head>
    <body>
        <div class="article-content">
            <h1>Introduction to Natural Language Processing</h1>
            <p class="author">By: Dr. John Doe</p>
            <p class="body-text">
                Text processing is a crucial step in <b>Machine Learning</b> and 
                <i>Artificial Intelligence</i>. It allows algorithms to understand unstructured data.
            </p>
            <a href="https://example.com/learn-more">Read More</a>
        </div>
    </body>
    </html>
    """

    # 1. Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # 2. Extract title text
    print("--- Document Title ---")
    print(soup.title.string)
    print("\n" + "=" * 40 + "\n")

    # 3. Extract specific tag content
    heading = soup.find("h1").text
    print("Heading (h1):", heading)

    author = soup.find("p", class_="author").text
    print("Author:", author)
    print("\n" + "=" * 40 + "\n")

    # 4. Strip HTML tags and extract clean body text
    body_paragraph = soup.find("p", class_="body-text").get_text(strip=True)
    print("Clean Body Text (Tags Stripped):")
    print(body_paragraph)
    print("\n" + "=" * 40 + "\n")

    # 5. Extract hyperlinks (href attribute)
    link = soup.find("a")["href"]
    print("Extracted URL:", link)
