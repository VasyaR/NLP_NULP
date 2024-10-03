import argparse
import pickle
from collections import Counter 

class NGramModel:

    def __init__(self, vocab: [str], n: int):
        self.n = n
        self.vocab = vocab
        self.unigram_counts = Counter()
        self.bigram_counts = Counter()
        self.trigram_counts = Counter()

    def train(self, tokens: [str]) -> None:

        tokens = [t for t in tokens if t in self.vocab]

        #Count unigrams
        self.unigram_counts = Counter(tokens)
        self.bigram_counts = Counter(zip(tokens, tokens[1:]))
        self.trigram_counts = Counter(zip(tokens, tokens[1:], tokens[2:]))

    def save(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self, f)



def tokenize(text: str) -> [str]:
    return text.split()


def train(args):
    with open(args.data) as f:
        lines = f.read().split('\n')
    
    tokens = []
    for line in lines:
        tokens += ["<s>"] + tokenize(line) + ["</s>"]

    counts = Counter(tokens)
    vocab = [token for token, count in counts.most_common(args.vocab_size)]  

    print(f"Vocab size: {len(vocab)}")
    
    model = NGramModel(vocab, args.n)
    model.train(tokens)
    model.save(args.model)

    print(f"Model saved to {args.model}")

def main():
    parser = argparse.ArgumentParser(description='N-gram LM')

    parser.add_argument("--data", required=True, help="Path to the data file")
    parser.add_argument("--model", required=True, help="Path to output model file")
    parser.add_argument("--n", type=int, default=2, help="N-gram order")
    parser.add_argument("--vocab-size", type=int, default=10_000, help="Vocabulary size")

    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()