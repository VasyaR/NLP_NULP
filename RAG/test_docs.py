import glob
import json
from tokenizing import tokenize_doc, tokenize_doc_to_str, tokenize_text
from rank_bm25 import BM25Okapi
import os
import pickle
import numpy as np
from tqdm import tqdm
import torch

docs = []
base_path = "./Data"
bm25_path = os.path.join(base_path, "bm25.pkl")
tokenized_docs_path = os.path.join(base_path, "tokenized_docs.pkl")

# Take all json files with names that end '_processed'  
for path in glob.glob(f"{base_path}/*_processed.json"):
    print(path)
    with open(path, 'r') as f:
        docs.extend(json.load(f))
    
index = 0

# for i, doc in enumerate(docs):
#     if 'body' in doc:
#         if doc['body'] == "I don't fuck the sandwich before eating it":
#             tokenized_doc = tokenize_doc(doc)
#             print(tokenized_doc)
#             index = i

with open(bm25_path, 'rb') as f:
    bm25 = pickle.load(f)

# tokenized_docs = [tokenize_doc(doc) for doc in tqdm(docs, desc="Tokenizing documents")]

# bm25 = BM25Okapi(tokenized_docs)

# with open(tokenized_docs_path, 'wb') as f:
#     pickle.dump(tokenized_docs, f)
# with open(bm25_path, 'wb') as f:
#     pickle.dump(bm25, f)

message = "tell me a joke about sandwich before eating it"
tokenized_message = tokenize_text(message)
print(tokenized_message)
scores = torch.tensor(bm25.get_scores(tokenized_message))
sorted_doc_indices = np.argsort(scores)

for i in range(1, 2):
    print("Score:", scores[sorted_doc_indices[-i]] )
    print(docs[sorted_doc_indices[-i]])
    print("Doc number:", sorted_doc_indices[-i])

# result_docs = [docs[i] for i in sorted_doc_indices[-30:] if scores[i] > 0]

# return result_docs[::-1] # Return the top n documents in descending order which means the most relevant documents are first
    