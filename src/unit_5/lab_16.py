import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize


def lab_16():
    """Lab 16: Natural Language Processing (NLP) Basics with NLTK

    This lab demonstrates fundamental NLP techniques using the NLTK library in Python.
    It covers tokenization, stop word removal, stemming, and lemmatization.

    Steps:
    1. Tokenize a sample text into sentences and words.
    2. Remove stop words and punctuation from the tokenized words.
    3. Apply stemming to reduce words to their root forms.
    4. Apply lemmatization to reduce words to their base dictionary forms.
    """
    # Download required NLTK corpuses and tokenizers (run once)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

    # Sample raw text corpus
    text = """
    Natural Language Processing (NLP) is a fascinating field of Artificial Intelligence. 
    It enables computers to understand, interpret, and manipulate human language. 
    Researchers are running multiple experiments to improve models!
    """

    print("--- 1. TOKENIZATION ---")
    # A. Sentence Tokenization (splitting text into individual sentences)
    sentences = sent_tokenize(text)
    print("Sentences count:", len(sentences))
    print("First Sentence:", sentences[0].strip())

    # B. Word Tokenization (splitting sentences into words/tokens)
    words = word_tokenize(text)
    print("\nTotal Tokens (Words + Punctuation):", len(words))
    print("First 15 Tokens:", words[:15])

    print("\n" + "=" * 50 + "\n")
    print("--- 2. STOP WORD REMOVAL ---")
    # Stop words are common words (like 'is', 'and', 'the', 'of') that carry little analytical value
    stop_words = set(stopwords.words("english"))

    # Filter out punctuation and stop words
    filtered_words = [
        w.lower() for w in words if w.isalnum() and w.lower() not in stop_words
    ]
    print("Filtered Words (No stop words or punctuation):")
    print(filtered_words[:12])

    print("\n" + "=" * 50 + "\n")
    print("--- 3. STEMMING VS LEMMATIZATION ---")
    # Sample words with different morphological forms
    sample_words = ["running", "runs", "ran", "easily", "fairly", "better"]

    # A. Stemming (Rule-based suffix stripping; often produces root stems that aren't real words)
    stemmer = PorterStemmer()
    stemmed_results = [stemmer.stem(w) for w in sample_words]
    print("Stemmed Words (PorterStemmer):", stemmed_results)

    # B. Lemmatization (Dictionary-based reduction to base/dictionary form 'lemma')
    lemmatizer = WordNetLemmatizer()
    # pos='v' specifies verb context for better lemmatization accuracy
    lemmatized_results = [lemmatizer.lemmatize(w, pos="v") for w in sample_words]
    print("Lemmatized Words (WordNetLemmatizer):", lemmatized_results)
