import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
       
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embed_size)

        ## TODO: define the LSTM
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, batch_first=True)
        
        ## TODO: define the final, fully-connected output layer
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
        
        self.hidden = (torch.zeros(1, 1, self.hidden_size),torch.zeros(1, 1, self.hidden_size))

       
    
    def forward(self, features, captions):
        # create embedded word vectors for each word in a sentence
        embeds = self.word_embeddings(captions[:,:-1])
        
        #Join batch of images/features with corresponding batch of embed captions.
        images_captions = torch.cat((features.unsqueeze(1),embeds),1)
       
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        lstm_out, self.hidden = self.lstm(images_captions)
        
        outputs = self.fc(lstm_out)
        
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        array = []
        count = 0
        wordidx = 0
        states = (torch.zeros(1, 1, self.hidden_size).to(inputs.device), torch.zeros(1, 1, self.hidden_size).to(inputs.device))
       
        while (count < max_len and wordidx != 1):
            lstm_out, states = self.lstm(inputs, states)
            outputs = self.fc(lstm_out).squeeze(1)
            _, wordidx = torch.max(outputs, dim = 1)
            array.append(wordidx.cpu().numpy()[0].item())
            inputs = self.word_embeddings(wordidx).unsqueeze(1) 
            count = count + 1
        return array
        