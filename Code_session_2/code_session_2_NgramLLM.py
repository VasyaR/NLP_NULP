import argparse
import pickle
import math
import pytest
import random
import numpy as np
import sys
from collections import Counter 

class NGramModel:

    def __init__(self, vocab: [str], n: int, smoothing: float = 0.0):
        self.n = n
        self.smoothing = smoothing
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

    def unigram_log_prob(self, token: str) -> float:

        count = self.unigram_counts.get(token, 0) + self.smoothing
        all_counts = sum(self.unigram_counts.values()) + self.smoothing * len(self.vocab)
        prob = count / all_counts
        return math.log(prob)     
    
    def bigram_log_prob(self, prev_token: str, token: str) -> float:

            bigram = (prev_token, token)
            bigram_count = self.bigram_counts.get(bigram, 0) + self.smoothing

            # My answer is wrong, because we are counting the probability of appearance second word adter the first
            # bigram_prob = bigram_count / sum(self.bigram_counts.values())
            # myans = math.log(bigram_prob) + self.unigram_log_prob(token)
            
            total = self.unigram_counts.get(prev_token, 0) + self.smoothing * len(self.vocab)

            # Lector answer
            prob = bigram_count / total 
            # print(f"My answer: {myans}, Lector answer: {math.log(prob)}")
            return math.log(prob)
    
    def trigram_log_prob(self, prev_prev_token: str, prev_token: str, token: str) -> float:

            trigram = (prev_prev_token, prev_token, token)
            trigram_count = self.trigram_counts.get(trigram, 0) + self.smoothing

            total = self.bigram_counts.get((prev_prev_token, prev_token), 0) + self.smoothing * len(self.vocab)

            prob = trigram_count / total
            return math.log(prob)
    


            
    
    def score_log_probs(self, tokens: [str]) -> float:

        log_probs = 0.0

        for i in range(len(tokens)):
            if i == 0:
                log_probs += self.unigram_log_prob(tokens[i])

            elif i == 1:
                log_probs += self.bigram_log_prob(tokens[i-1], tokens[i])
            
            else: 
                log_probs += self.trigram_log_prob(tokens[i-2], tokens[i-1], tokens[i])
            
        return log_probs
        # the cat sat on the mat

    def sample(self, prefix: [str]) -> str:
        
        scores = []
        probs = []
        for token in self.vocab:
            score = self.score_log_probs(prefix + [token])
            prob = math.exp(score)
            scores.append((score, token))
            probs.append(prob)
        
        probs = np.array(probs)
        probs /= probs.sum()

        return np.random.choice(self.vocab, p=probs)  

    def generate(self):
        pass

    def save(self, filename: str) -> None:
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str) -> "NGramModel":
        with open(filename, 'rb') as f:
            return pickle.load(f)



def tokenize(text: str) -> [str]:
    return text.split()

def tokenize_dataset(dataset_path: str) -> [str]:
    with open(dataset_path) as f:
        lines = f.read().split('\n')
    
    tokens = []
    for line in lines:
        tokens += ["<s>"] + tokenize(line) + ["</s>"]
    
    return tokens


def train(args):
    
    tokens = tokenize_dataset(args.data)

    counts = Counter(tokens)
    vocab = [token for token, count in counts.most_common(args.vocab_size)]  

    print(f"Vocab size: {len(vocab)}")
    
    model = NGramModel(vocab, args.n, smoothing=args.smoothing)
    model.train(tokens)
    model.save(args.model)

    print(f"Model saved to {args.model}")

def score(args):
    
    model = NGramModel.load(args.model)

    data = tokenize_dataset(args.data)

    n_log_probs = -model.score_log_probs(data)

    perplexity = math.exp(n_log_probs/len(data))

    print(f"Negative Log probs = {n_log_probs}")
    print(f"Perplexity = {perplexity}")

def generate(args):
    model = NGramModel.load(args.model)

    prefix = ["<s>"]
    for i in range(100):
        token = model.sample(prefix)
        prefix.append(token)

        if token == "</s>":
            print("EOF")
            # break

    else:
        print(token, end=" ")
        sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser(description='N-gram LM')

    # add subparsers for train and score
    subparser = parser.add_subparsers(help='sub-command help', dest="command")

    # train subparser
    parser_train = subparser.add_parser('train', help='train help') 
    parser_train.add_argument("--data", required=True, help="Path to the data file")
    parser_train.add_argument("--model", required=True, help="Path to output model file")
    parser_train.add_argument("--n", type=int, default=2, help="N-gram order")
    parser_train.add_argument("--vocab-size", type=int, default=10_000, help="Vocabulary size")
    parser_train.add_argument("--smoothing", type=float, default=0.1, help="Laplasian smoothing")

    # score subparser
    parser_score = subparser.add_parser('score', help='score help')

    parser_score.add_argument("--model", required=True, help="Path to the model file")
    parser_score.add_argument("--data", required=True, help="Path to the data file")

    # generate subparser
    parser_generate = subparser.add_parser('generate', help='generate help')

    parser_generate.add_argument("--model", required=True, help="Path to the model file")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "score":
        score(args) 
    elif args.command == "generate":
        generate(args)

def test_unigram_log_prob():
    doc = ["the", "cat", "sat", "on", "the", "mat"]
    vocab = ["cat", "mat", "on", "sat", "the"]

    model = NGramModel(vocab, 1)
    model.train(doc)

    assert model.unigram_log_prob("cat") == math.log(1 / 6)
    assert model.unigram_log_prob("the") == math.log(1 / 3)

def test_bigram_log_prob():
    doc = ["the", "cat", "sat", "on", "the", "mat"]
    vocab = ["cat", "mat", "on", "sat", "the"]

    model = NGramModel(vocab, 2)
    model.train(doc)

    assert model.bigram_log_prob("the", "cat") == math.log(1 / 2)
    with pytest.raises(Exception):
        assert model.bigram_log_prob("cat", "the") == math.log(0)

def test_bigram_log_prob_with_smoothing():
    doc = ["the", "cat", "sat", "on", "the", "mat"]
    vocab = ["cat", "mat", "on", "sat", "the"]

    model = NGramModel(vocab, 2, smoothing=0.1)
    model.train(doc)

    assert model.bigram_log_prob("the", "cat") == math.log(1.1 / 2.5)
    assert model.bigram_log_prob("cat", "the") == math.log(0.1 / 1.5)

def test_trigram_log_prob():
    doc = ["the", "cat", "sat", "on", "the", "mat"]
    vocab = ["cat", "mat", "on", "sat", "the"]

    model = NGramModel(vocab, 3)
    model.train(doc)

    assert model.trigram_log_prob("the", "cat", "sat") == math.log(1 / 1)
    with pytest.raises(Exception):
        assert model.trigram_log_prob("cat", "the", "sat") == math.log(0)


if __name__ == '__main__':
    main()