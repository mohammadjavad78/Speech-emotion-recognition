
import torch.nn as nn
from transformers import  HubertModel

class HUBERTClassification(nn.Module):
    def __init__ (self,meanonnotzero=False):
        super(HUBERTClassification, self).__init__()
        self.bert = HubertModel.from_pretrained("superb/hubert-base-superb-ks")
        self.bert.feature_extractor._freeze_parameters()
        self.bat2 = nn.BatchNorm1d(768)
        self.rel2 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(0.4)
        self.out2 = nn.Linear(768, 6)
        self.rel3 = nn.ReLU(inplace=True)
        self.meanonnotzero=meanonnotzero

    def forward(self,inputs):
        inputs=inputs[:,0]
        output = self.bert(inputs)['last_hidden_state']
        del inputs
        if(self.meanonnotzero):
            output = self.bat2(output.mean(axis=1))      #for wavtovec
        else:
            outputs = output.sum(dim=1) / (output!=0).sum(dim=1) #for wavtovec for not zeros
            output = self.bat2(outputs)      #for wavtovec for not zeros
        output = self.rel2(output)
        output = self.drop1(output)
        output = self.out2(output)
        output = self.rel3(output)


        return output
    
# print(BERTClassification())
    