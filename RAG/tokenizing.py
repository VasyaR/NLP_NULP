import nltk
from nltk.tokenize import word_tokenize

# Download the necessary NLTK data files
nltk.download('punkt')

def tokenize_text(text: str) -> list:
    # Convert text to lowercase before tokenization
    text = text.lower()
    return word_tokenize(text)

def tokenize_doc(doc: dict) -> list:
    tokenized_doc = []
    for key, value in doc.items():
        tokenized_key = key.lower().replace("_", " ")
        tokenized_doc.append(tokenized_key)
        tokenized_doc.append(':')
        if isinstance(value, str):
            tokenized_doc.extend(tokenize_text(value))
        else:
            tokenized_doc.extend(tokenize_text(str(value)))
    return tokenized_doc

def tokenize_doc_to_str(doc: dict) -> str:
    tokenized_doc = []
    for key, value in doc.items():
        tokenized_key = key.lower().replace("_", " ")
        tokenized_doc.append(tokenized_key)
        tokenized_doc.append(':')
        if isinstance(value, str):
            tokenized_doc.extend(value)
        else:
            tokenized_doc.extend(str(value))
    return ' '.join(tokenized_doc)

# Example usage
user_message = "Tell me a joke about computers."
tokenized_message = tokenize_text(user_message)
# print(tokenized_message)

doc = {
    "title": "Funny Computer Joke",
    "content": "Why do programmers prefer dark mode? Because light attracts bugs!",
    "rating": 5
}
tokenized_doc = tokenize_doc(doc)
# print(tokenized_doc)