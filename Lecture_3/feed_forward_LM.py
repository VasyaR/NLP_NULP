import torch
from torch import nn
from typing import Iterable
from tqdm import tqdm

class Vocabulary:

  def __init__(self, tokens, unk_token="<unk>"):
    self.unk_token = unk_token
    self.unk_index = 0
    self._itos = set([unk_token] + tokens)
    self._stoi = {token: index for index, token in enumerate(self._itos)}

  def stoi(self, token: str) -> int:
    """Return token index or `<unk>` index if `token` is not in the vocab.
    """
    return self._stoi.get(token, self.unk_index)


  def itos(self, index: int) -> str:
    """Return token by its `index`.

    Raise LookupError if `index` is out of vocabulary range.
    """

    return self._itos[index]

  @property
  def tokens(self):
    return self._itos

  def __len__(self) -> int:
    return len(self._itos)


class BengioLMModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, context_len: int, hidden_dim: int) -> None:

        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.context_len = context_len
        self.hidden_dim = hidden_dim
         
        self.embed = nn.Embedding(vocab_size, embed_dim) # vocab_size * embed_dim
        self.W = nn.Linear(context_len * embed_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.U = nn.Linear(hidden_dim, vocab_size)

    def forward(self, X_indexes: torch.tensor):

        """
        
        Args:
            X_indexes: tensor of indexes of context tokens.
        """

        X = self.embed(X_indexes) # [batch_size, context len * embed dim]
        e = X.view(-1, self.context_len * self.embed_dim)

        h = self.tanh(self.W(e))

        logits = self.U(h)

        log_probs = torch.log_softmax(logits, dim=-1)

        return log_probs

def tokenize(text: str) -> [str]:
   return list(text.lower())

def batch_it(xs, batch_size):
    batch = []

    for i, x in enumerate(xs):
        batch.append(x)

        if i % batch_size == batch_size - 1:
            yield batch
            batch = []

    if batch:
        yield batch

def test_batch_it():
    xs = list("abcdefgh") 
    batch_size = 3
    expected = [["a", "b", "c"], ["d", "e", "f"], ["g", "h"]]

    actual = list(batch_it(xs, batch_size))
    print(actual)

    assert actual == expected 


def train():

    train_text = open("/home/beav3r/semestr7vsl/NLP/Code_session_2/data/train.txt").read()
    train_text = train_text[:1000]

    train_text_tokens = tokenize(train_text)
    vocab = Vocabulary(train_text_tokens)
    print(len(vocab))

    hparams = {
        "vocab_size": len(vocab),
        "embed_dim": 64,
        "context_len": 4,
        "hidden_dim": 128,
        "learning_rate": 0.001,
        "num_epochs": 25,
        "batch_size": 256
    }

    model = BengioLMModel(vocab_size=hparams["vocab_size"], embed_dim=hparams["embed_dim"], 
                          context_len=hparams["context_len"], hidden_dim=hparams["hidden_dim"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"])
    loss_fn = nn.NLLLoss()

    for epoch in range(hparams["num_epochs"]):
        total_loss = 0.0
        examples = prepare_data(train_text_tokens, hparams["context_len"])
        examples = list(examples)
        
        for batch in tqdm(batch_it(examples, batch_size=hparams["batch_size"]), leave=False):


            X_batch = []
            y_batch = []
            for context, target in batch:
                X = vectorize(context, vocab)
                y = vectorize([target], vocab)
                X_batch.append(X)
                y_batch.append(y)

            X_batch = torch.stack(X_batch)
            y_batch = torch.tensor(y_batch)

            optimizer.zero_grad()

            log_probs = model(X_batch)
            loss = loss_fn(log_probs, y_batch)

            loss.backward()
            optimizer.step()
            total_loss += loss.sum().item()

        print(f" Epoch: {epoch} Loss: {total_loss / len(examples)}")
    
    # save model
    torch.save({"model_state_dict": model.state_dict(),
                "vocab": vocab,
                "hparams": hparams}, "model.pt")
       

def vectorize(tokens: Iterable[str], vocab: Vocabulary) -> torch.tensor:
   X = torch.tensor([vocab.stoi(token) for token in tokens]) 
   return X

def prepare_data(tokens: [str], context_len: int) -> [([str], str)]:
    """

    Args:
        tokens: list of tokens
        context_len: length of context

    Reurns:
        Iterable of (context, target) pairs
    """
    # res = []

    for i in range(context_len, len(tokens)):
        context = tokens[i - context_len:i]
        target = tokens[i]

        yield (context, target)

def test_prepare_data():
   
   tokens = ["the", "students", "opened", "their", "books"]
   context_len = 4

   excepted = [(["the", "students", "opened", "their"], "books")]

   actual = list(prepare_data(tokens, context_len))

   assert actual == excepted

def test_prepare_data_context_len_3():
   
   tokens = ["the", "students", "opened", "their", "books"]
   context_len = 3

   excepted = [(["the", "students", "opened"], "their"),
               (["students", "opened", "their"], "books")]

   actual = list(prepare_data(tokens, context_len))

   assert actual == excepted

if __name__ == "__main__":
    
    train()
