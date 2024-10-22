import torch
import json

class RNN():
    def __init__(self, embeddings, W_h, W_y, U_h, vocab):
        self.W_h = W_h
        self.W_y = W_y
        self.U_h = U_h
        self.embeddings = embeddings
        self.vocab = vocab

    def forward(self):
        ans = ["["] 
        prev_char = "["
        h_prev = torch.zeros(256)
        while prev_char != "]":
            x_t = self.embeddings[self.vocab.index(prev_char)]
            h_t = torch.tanh(self.W_h @ x_t  + self.U_h @ h_prev)
            h_prev = h_t
            y_t = torch.softmax(self.W_y @ h_t, dim=-1)
            prev_char = self.vocab[torch.argmax(y_t).item()]
            ans.append(prev_char)

        return ans
    
def main():

    with open("vocab.json", "r") as f:
        vocab = json.load(f)
    
    with open("embedding.weight.json", "r") as f:
        embeddings = torch.tensor(json.load(f))
    
    with open("W_h.weight.json", "r") as f:
        W_h = torch.tensor(json.load(f))
    
    with open("W_y.weight.json", "r") as f:
        W_y = torch.tensor(json.load(f))
   
    with open("U_h.weight.json", "r") as f:
        U_h = torch.tensor(json.load(f))

    rnn = RNN(embeddings, W_h, W_y, U_h, vocab)

    ans_token_list = rnn.forward()
    ans = "".join(ans_token_list)
    print("Answer: ", ans)

if __name__ == "__main__":
    main()

# Vocab shape: 133
# Embedding shape: torch.Size([133, 96])
# W_h shape: torch.Size([256, 96])
# W_y shape: torch.Size([133, 256])
# U_h shape: torch.Size([256, 256])

