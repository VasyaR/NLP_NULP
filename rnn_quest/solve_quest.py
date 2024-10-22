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
        
    def train(self, message, epochs=1000, lr=10.01):
        optimizer = torch.optim.Adam([self.W_h, self.W_y, self.U_h], lr=lr)
        self.U_h.requires_grad_(True)
        self.W_h.requires_grad_(True)
        self.W_y.requires_grad_(True)
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = 0
            prev_char = message[0]
            h_prev = torch.zeros(256).requires_grad_(True)
            for char in message[1:]:
                x_t = self.embeddings[self.vocab.index(prev_char)]
                h_t = torch.tanh(self.W_h @ x_t  + self.U_h @ h_prev)
                h_prev = h_t
                y_t = torch.softmax(self.W_y @ h_t, dim=-1)
                loss += 1 - y_t[self.vocab.index(char)]
                prev_char = char
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print("Epoch: ", epoch, "Loss: ", loss.item())
            
        # Save the weights, rewrite if already exists
        with open("W_h_new.weight.json", "w") as f:
            json.dump(self.W_h.tolist(), f)

        with open("W_y_new.weight.json", "w") as f:
            json.dump(self.W_y.tolist(), f)

        with open("U_h_new.weight.json", "w") as f:
            json.dump(self.U_h.tolist(), f)
        

        
    
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

    # ans_token_list = rnn.forward()
    # ans = "".join(ans_token_list)
    # print("Answer: ", ans)

    message = "[NARUTO]"

    rnn.train(message, epochs=1000, lr=0.01)
    ans_token_list = rnn.forward()
    ans = "".join(ans_token_list)
    print("New Answer: ", ans)

def test():
    with open("vocab.json", "r") as f:
        vocab = json.load(f)
    
    with open("embedding.weight.json", "r") as f:
        embeddings = torch.tensor(json.load(f))
    
    with open ("U_h_new.weight.json", "r") as f:
        U_h = torch.tensor(json.load(f))
    
    with open("W_h_new.weight.json", "r") as f:
        W_h = torch.tensor(json.load(f))
    
    with open("W_y_new.weight.json", "r") as f:
        W_y = torch.tensor(json.load(f))
    
    rnn = RNN(embeddings, W_h, W_y, U_h, vocab)
    ans_token_list = rnn.forward()
    ans = "".join(ans_token_list)
    print("New Answer: ", ans)

if __name__ == "__main__":
    # main()
    test()

# Vocab shape: 133
# Embedding shape: torch.Size([133, 96])
# W_h shape: torch.Size([256, 96])
# W_y shape: torch.Size([133, 256])
# U_h shape: torch.Size([256, 256])

