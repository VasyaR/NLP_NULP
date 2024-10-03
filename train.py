training_text = """
Algorithms using artificial intelligence are discovering unexpected tricks to solve problems that astonish their developers. But it is also raising concerns about our ability to control them.

The gaggle of Google employees peered at their computer screens in bewilderment. They had spent many months honing an algorithm designed to steer an unmanned hot air balloon all the way from Puerto Rico to Peru. But something was wrong. The balloon, controlled by its machine mind, kept veering off course.
Salvatore Candido of Google's now-defunct Project Loon venture, which aimed to bring internet access to remote areas via the balloons, couldn't explain the craft’s trajectory. His colleagues manually took control of the system and put it back on track.
It was only later that they realised what was happening. Unexpectedly, the artificial intelligence (AI) on board the balloon had learned to recreate an ancient sailing technique first developed by humans centuries, if not thousands of years, ago. "Tacking" involves steering a vessel into the wind and then angling outward again so that progress in a zig-zag, roughly in the desired direction, can still be made.
"""
import re
import torch


CONTEXT_SIZE = 2
NUM_EPOCH = 10


Token = str
Context = list[Token]
Target = Token


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text)


def preprocess(text):
    text = text.lower()
    tokens = tokenize(text)
    return tokens


def make_training_examples(
    tokens: list[str], context_size: int
) -> tuple[Context, Target]:
    for i in range(0, len(tokens) - context_size):
        left_context = tokens[i : i + context_size]
        target = tokens[i + context_size]
        right_context = tokens[i + context_size + 1 : i + 2 * context_size + 1]
        context = left_context + right_context
        yield context, target


class Vocabulary:
    def __init__(self, tokens: list[str]) -> None:
        self.vocab = sorted(set(tokens))
        self.token_to_index = {token: i for i, token in enumerate(self.vocab)}
        self.index_to_token = {i: token for i, token in enumerate(self.vocab)}


def vectorize(
    context: Context, target: Target, vocab: Vocabulary
) -> tuple[list[int], int]:
    context_vector = [vocab.token_to_index[token] for token in context]
    target_vector = vocab.token_to_index[target]
    return context_vector, target_vector


def main():
    tokens = preprocess(training_text)
    vocab = Vocabulary(tokens)

    V = vocab_size = len(vocab.vocab)
    N = embed_dim = 16

    W_in = torch.randn(V, N, requires_grad=True)
    W_out = torch.randn(N, V, requires_grad=True)

    optimizer = torch.optim.SGD([W_in, W_out], lr=0.01)

    for epoch in range(NUM_EPOCH):
        epoch_loss = 0
        num_examples = 0
        for training_example in make_training_examples(tokens, CONTEXT_SIZE):
            num_examples += 1
            optimizer.zero_grad()
            context, target = vectorize(*training_example, vocab)
            hidden = torch.zeros(N)
            for c in context:
                hidden += W_in[c]

            logits = hidden @ W_out  # (vocab_size)
            log_probs = torch.log_softmax(logits, dim=0)

            loss = -log_probs[target]  # negative log likelihood (NLLLoss)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss = epoch_loss / num_examples
        print(f"Epoch: {epoch}, Loss: {epoch_loss}")

    embeddings = W_in + W_out.t()


if __name__ == "__main__":
    main()
