from litellm import completion
import os
from prompt import PROMPT

from retriever import Retriever

import glob
import json

# env variable
# os.environ['GROQ_API_KEY'] = ""

# Define the base path
base_path = "./Data" # "/mnt/d/Semester7/NLP/RAG/Data"

class JokeGeneratorBot:

    def __init__(self, docs: [str], PROMPT: str) -> None:
        self.retriever = Retriever(docs)
        self.PROMPT = PROMPT

    def generate_joke(self, user_message, n=30, bm25_only=False, semantic_only=False, scores_combination=True, bm_koef=0.75):

        context = self.retriever.get_docs(user_message, n=n, bm25_only=bm25_only, semantic_only=semantic_only, scores_combination=scores_combination, bm_koef=bm_koef)
        
        messages = [
        {"role": "system", "content": self.PROMPT},
        {"role": "user", "content": f"Context:\n{context}\nUser message: {user_message}"},
    ]

        response = completion(
        model="groq/llama-3.3-70b-versatile", 
        messages=messages,
    )
        print("Context:", context)

        return  {"Response": response.choices[0].message.content, "Context": context}

def main():
    user_message = "Tell me a joke about Whales and a seaman"

    # Take all json files with names that end '_processed' 
    docs = []
    for path in glob.glob(f"{base_path}/*_processed.json"):
        with open(path, 'r') as f:
            docs.extend(json.load(f))
    
    bot = JokeGeneratorBot(docs, PROMPT)
    responce_and_context = bot.generate_joke(user_message)

    print(f"User message:", {user_message})
    print(f"Answer:", {responce_and_context["Response"]})

    return

if __name__ == "__main__":
    main()
