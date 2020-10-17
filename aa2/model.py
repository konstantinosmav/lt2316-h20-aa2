import torch
import torch.nn as nn

num_layers = 3
emb_size = 100 #around 100 seems like a good number of dimensions for the word embeddings
out_size = 5 #num of ner labels


class MyRecurrentNet(nn.Module) :
    def __init__(self, vocab_size, hidden_size, num_layers,emb_size,out_size, bidirectional) :
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.emb_size = emb_size
        
        #embeddings layer
        self.embeddings = nn.Embedding(vocab_size, 
                                       emb_size)
        
        
       #lstm layer 
        self.lstm = nn.LSTM(emb_size,
                         hidden_size,
                         num_layers)
        
        # Output linear layer
        self.classifier = nn.Linear(hidden_size,
                                    out_size,
                                    batch_first= True,                                    
                                    bias=False)
        
    def forward(self, sent):
        # RNN returns output and hidden state
        embeds = torch.sum(embeddings(sent), dim=2)        
        
        
        lstm_out, (h, c) = self.lstm(embeds) # h and c are the hidden layers and their size is (num_layers, number of elements, hidden_size)
        
     
        
        # Output layer
       
        prediction = self.classifier(lstm_out)
        return prediction, (h,c)
    