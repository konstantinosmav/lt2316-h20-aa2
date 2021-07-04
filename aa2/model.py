import torch
import torch.nn as nn


class ClassifierModel(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, output_size, num_layers):
        super(ClassifierModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedd = nn.Embedding(input_size, emb_size)
        self.rnn = nn.LSTM(emb_size, hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        
        self.predict = nn.Linear(hidden_size, output_size)

    def forward(self, sentence, hidden_st):
        #torch.Size([25, 95, 1]) original size
        #torch.Size([2375, 1]) transfromed with view
        #torch.Size([2375, 1, 100]) embedding size
        #torch.Size([25, 95, 1, 100]) second embedding size
        #torch.Size([25, 95, 100]) after summation, third embedding size
        # need to convert the 3d tensor of shape BSWsize to a 2d tensor of shape BSxW and give it to the embedding layer.
        emb = self.embedd(sentence.view(-1, sentence.size(2)))
        # convert the resulting 3d tensor to a 4d tensor 
        emb = emb.view(*sentence.size(), -1)
        emb = emb.sum(2)

        out, hidden_st = self.rnn(emb, hidden_st)
        scores = self.predict(out)
        
        return scores, hidden_st

    def init_hidden(self, batch_size):

        w = next(self.parameters()).data

        
        hidden_st = (w.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                     w.new(self.num_layers, batch_size, self.hidden_size).zero_())

        return hidden_st
  
        
