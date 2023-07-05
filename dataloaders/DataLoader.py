import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def changetype(i):
  if(i=='S'):
    return 0
  elif(i=='A'):
    return 1
  elif(i=='H'):
    return 2
  elif(i=='W'):
    return 3
  elif(i=='F'):
    return 4
  elif(i=='N'):
    return 5
  return 6

class MyDataset(Dataset):
    def __init__(self,train=True,test=False,val=False,dir='./drive/MyDrive/datasets',rand=5):
        self.dir = dir
        self.train_path_list=[]
        self.train_path_list_label=[]
        self.train=train
        self.test=test
        self.val=val
        self.rand=rand
        super(MyDataset, self).__init__()

        for top, _, files in os.walk(os.path.join(self.dir,'male')):
            if(files!=[]):
                for j in files:
                    self.train_path_list.append(os.path.join(top,j))
                    # print(j)
                    status=j.split('/')[-1].split('.')[0][-3]
                    status=changetype(status)
                    self.train_path_list_label.append(status)

        for top, _, files in os.walk(os.path.join(self.dir,'female')):
            if(files!=[]):
                for j in files:
                    self.train_path_list.append(os.path.join(top,j))
                    status=j.split('/')[-1].split('.')[0][-3]
                    status=changetype(status)
                    self.train_path_list_label.append(status)
        self.imgs=pd.DataFrame.from_dict({'Address':self.train_path_list,'Labels':self.train_path_list_label})


        self.imgs = self.imgs[self.imgs['Labels'].duplicated(keep=False)]


        trainn, x = train_test_split(self.imgs,train_size=0.8,stratify=self.imgs['Labels'],random_state=rand)
        

      

        self.imgs2=pd.DataFrame.from_dict({'Address':x['Address'],'Labels':x['Labels']})
        self.imgs2 = self.imgs2[self.imgs2['Labels'].duplicated(keep=False)]


        testt, vall = train_test_split(self.imgs2,test_size = 0.5,train_size =0.5,stratify=self.imgs2['Labels'],random_state=rand)


        if(self.val==True):
            self.imgs=pd.DataFrame.from_dict({'Address':vall['Address'],'Labels':vall['Labels']})
        elif(self.test==True):
            self.imgs=pd.DataFrame.from_dict({'Address':testt['Address'],'Labels':testt['Labels']})
        else:
            self.imgs=pd.DataFrame.from_dict({'Address':trainn['Address'],'Labels':trainn['Labels']})




    def __getitem__(self, index):
        iloc=self.imgs.iloc[index]
        return iloc
        

    def __len__(self):
        return len(self.imgs)
    

from transformers import Wav2Vec2FeatureExtractor,HubertModel
import soundfile as sf
import os
import librosa    
processor = Wav2Vec2FeatureExtractor.from_pretrained("superb/hubert-base-superb-ks")


def map_to_array(batch):
    speech, sampling_rate = librosa.load(batch['Address'],sr=16000,duration=6)
    return speech,sampling_rate


def changed(address,maxlen=-1):
  speech,sampling_rate=map_to_array(address)
  if(maxlen==-1):
    input_values = processor(speech,sampling_rate=sampling_rate, return_tensors="pt").input_values 
  else:
    input_values = processor(speech,sampling_rate=sampling_rate,max_length=maxlen, padding='max_length',return_tensors="pt").input_values 
  del speech
  return input_values
    


def collate_fn(samples):
  maxlen=0
  editedsamples=[]
  editedsamples2=[]
  for i in samples:
    k=changed(i).shape
    if(maxlen<k[1]):
      maxlen=k[1]
    del k
  for i in samples:
    editedsamples.append(changed(i,maxlen))
    status=i['Labels']
    editedsamples2.append(torch.tensor(status))
  batch = torch.stack(editedsamples, dim=0)
  batch2 = torch.stack(editedsamples2, dim=0)
  del editedsamples
  del editedsamples2
  return batch,batch2



