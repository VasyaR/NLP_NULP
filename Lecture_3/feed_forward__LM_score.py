import torch
from feed_forward_LM import BengioLMModel, Vocabulary, tokenize, prepare_data, vectorize

def load_model(path):
    state = torch.load(path)
    hparams = state["hparams"]
    vocab = state["vocab"]

    model = BengioLMModel(vocab_size=hparams["vocab_size"],
                        embed_dim=hparams["embed_dim"],
                        context_len=hparams["context_len"],
                        hidden_dim=hparams["hidden_dim"])

    model.load_state_dict(state["model_state_dict"])

    return model, vocab

def compute_score(model: BengioLMModel, sentence: str, vocab:Vocabulary) -> float:
    
    tokens = tokenize(sentence)

    total_log_prob = 0.0

    for context, target in prepare_data(tokens, model.context_len):
        
        X = vectorize(context, vocab)
        target = vectorize([target], vocab)[0]
        log_probs = model(X)
        target_log_prob = log_probs[0, target]
        total_log_prob += target_log_prob

    return torch.exp(-total_log_prob / len(tokens)).item()

def run_score():
    model, vocab = load_model("/home/beav3r/semestr7vsl/NLP/Lecture_3/model.pt")

    test = ["Привіт світ!", "Приліт світ!"]

    for sentence in test:
        score = compute_score(model, sentence, vocab)
        print(sentence, score)

if __name__ == "__main__":
    run_score()