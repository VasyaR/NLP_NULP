from litellm import completion
import os
from prompt import PROMPT

from retriever import Retriever

import glob
import json

# env variable
os.environ['GROQ_API_KEY'] = "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"

# Define the base path
base_path = "/mnt/d/Semester7/NLP/RAG/Data"

class JokeGeneratorBot:

    def __init__(self, docs: [str], PROMPT: str) -> None:
        self.retriever = Retriever(docs)
        self.PROMPT = PROMPT

    def generate_joke(self, user_message):

        context = self.retriever.get_docs(user_message, n=40, semantic_only=True)
        # print(f"Context:\n{context}")
        messages = [
        {"role": "system", "content": self.PROMPT},
        {"role": "user", "content": f"Context:\n{context}\nUser message: {user_message}"},
    ]

        response = completion(
        model="groq/llama3-8b-8192", 
        messages=messages,
    )

        return  response.choices[0].message.content

def main():
    user_message = "Tell me a joke about Whales and a seaman"

    # Take all json files with names that end '_processed' 
    docs = []
    for path in glob.glob(f"{base_path}/*_processed.json"):
        with open(path, 'r') as f:
            docs.extend(json.load(f))
    
    bot = JokeGeneratorBot(docs, PROMPT)
    answer = bot.generate_joke(user_message)

    print(f"User message:", {user_message})
    print(f"Answer:", {answer})

    return

if __name__ == "__main__":
    main()
