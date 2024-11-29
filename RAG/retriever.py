from rank_bm25 import BM25Okapi
from tokenizing import tokenize_text, tokenize_doc, tokenize_doc_to_str
import numpy as np
from tqdm import tqdm
import os
import pickle
import torch

from sentence_transformers import SentenceTransformer

# Define the base path
base_path = "./Data" # "/mnt/d/Semester7/NLP/RAG/Data"

class Retriever:

    def __init__(self, docs: [dict]) -> None:
        self.docs = docs
        self.tokenized_docs_path = os.path.join(base_path, "tokenized_docs.pkl")
        self.bm25_path = os.path.join(base_path, "bm25.pkl")
        self.sbert_embeddings_path = os.path.join(base_path, "embeddings_parts")

        # Initialize SBERT
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        sbert = SentenceTransformer('sentence-transformers/all-distilroberta-v1', device=device)

        # Load or tokenize documents
        if os.path.exists(self.tokenized_docs_path) and os.path.exists(self.bm25_path) and os.path.exists(self.sbert_embeddings_path):
            print("Loading cache")
            self.load_cache()
            print("Cache loaded")
        else:
            self.tokenize_and_initialize()

    def tokenize_and_initialize(self):
        # Tokenize the documents with a progress bar
        self.tokenized_docs = [tokenize_doc(doc) for doc in tqdm(self.docs, desc="Tokenizing documents")]
        self.str_docs = [tokenize_doc_to_str(doc) for doc in self.docs]

        # Ensure that the tokenized_docs list is not empty
        if not self.tokenized_docs:
            raise ValueError("The list of tokenized documents is empty. Please check the input documents and the tokenization process.")

        # Initialize BM25 with the tokenized texts
        self.bm25 = BM25Okapi(self.tokenized_docs)

        # Get embeddings for all the tocenized documents
        self.sbert_embeddings = self.sbert.encode(self.str_docs, show_progress_bar=True)
        self.sbert_embeddings = self.sbert_embeddings.cpu()

        # Save the tokenized documents and BM25 model
        self.save_cache()

    def save_cache(self):
        with open(self.tokenized_docs_path, 'wb') as f:
            pickle.dump(self.tokenized_docs, f)
        with open(self.bm25_path, 'wb') as f:
            pickle.dump(self.bm25, f)
        split_size = 1000  # Number of rows per split
        embeddings_size = self.sbert_embeddings.size(0)

        # Split and save the tensor
        for i in range(0, embeddings_size, split_size):
            end_idx = min(i + split_size, embeddings_size)
            part = self.sbert_embeddings[i:end_idx].clone()
            torch.save(part, os.path.join(self.sbert_embeddings_path, f"embeddings_part_{i//split_size}.pt"))

    def load_cache(self):
        with open(self.tokenized_docs_path, 'rb') as f:
            self.tokenized_docs = pickle.load(f)
        with open(self.bm25_path, 'rb') as f:
            self.bm25 = pickle.load(f)
        print("Loading SBERT embeddings")
        # Load and combine
        loaded_parts = []
        files = os.listdir(self.sbert_embeddings_path)

        # Sort numerically based on the numeric part of the filename
        sorted_files = sorted(files, key=lambda x: int(x.split('_')[2].split('.')[0]))
        counter = 0
        for file in sorted_files:
            if file.startswith("embeddings_part_") and file.endswith(".pt"):
                part_path = os.path.join(self.sbert_embeddings_path, file)
                loaded_parts.append(torch.load(part_path))
                if counter % 50 == 0:
                    print("Loaded", file)
                counter += 1

        self.sbert_embeddings = torch.cat(loaded_parts, dim=0)
        print("SBERT embeddings loaded")

    def get_docs(self, user_message: str, n: int = 30, bm25_only: bool = False, semantic_only: bool = False, scores_combination: bool = True, bm_koef: float = 0.75) -> [str]:
        # In case of BM25 only, return the top n documents based on BM25 scores, if somebody sets a couple
        # of flags to True, the func will return the top n documents based on the first flag set to True

        if bm25_only:
            semantic_only = False
            scores_combination = False
            print("BM25 only")
            scores = torch.tensor(self._get_bm25_scores(user_message))
         
        elif semantic_only:
            scores_combination = False
            print("Semantic only")
            scores = self.get_semantic_scores(user_message)

        elif scores_combination:
            print("Combination")
            bm_scores = self._get_bm25_scores(user_message)
            semantic_scores = self.get_semantic_scores(user_message)
            scores = torch.tensor(bm_koef * bm_scores) + (1 - bm_koef) * semantic_scores

        # Sort the documents by their BM25 scores in descending order
        sorted_doc_indices = np.argsort(scores)

        result_docs = [self.docs[i] for i in sorted_doc_indices[-n:] if scores[i] > 0]

        return result_docs[::-1] # Return the top n documents in descending order which means the most relevant documents are first
    
    def _get_bm25_scores(self, user_message: str) -> np.array:
        tokenized_user_message = tokenize_text(user_message)
        return self.bm25.get_scores(tokenized_user_message)
    
    def get_semantic_scores(self, user_message: str) -> np.array:
        user_message_embedding = self.sbert.encode(user_message)
        scores = self.sbert.similarity(user_message_embedding, self.sbert_embeddings)
        return scores[0]