from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch
from TorchCRF import CRF
# In[数值初始化]
junior_max_length=-1
embedding_dim=100
hidden_dim=5
num_layers=10
batch_size=64
w2idx={}
# In[字典计算]
target_to_num={'O':0,'o':0,'B-MAT':1,'I-MAT':2}
# In[读入数据]
def read_file(root_dir):
    file = open(root_dir,mode='r',encoding='utf-8')
    file.seek(0,0)
    word=[]
    target=[]
    sentence=[]
    sen_target=[]
    for line in file:
        temp=line.split(" ")
        if len(temp)==2:
            word.append(temp[0])
            target.append(temp[1].strip())
        else:
            if len(word)>0:
                sentence.append(word)
                sen_target.append(target)
                word=[]
                target=[]
    return sentence,sen_target
# In[计算数组长度]
def c_maxlength(list1,max_length=-1):
    for line in list1:
        if(len(line)>max_length):
            max_length=len(line)
    return max_length
# In[计算词典]
def c_w2idx(list1,w2idx):
    for line in list1:
        for word in line:
           if word not in w2idx:
               w2idx[word]=len(w2idx)+1
# In[测试集与数据集的路径]
train_junior_root = "D:\\毕业设计\\data\\初中数学\\train_data"
test_junior_root = "D:\\毕业设计\\data\\初中数学\\test_data"
# In[导出数据]
train_junior_sentence,train_junior_target =  read_file(train_junior_root)
test_junior_sentence,test_junior_target = read_file(test_junior_root)
# In[对数值进行初始化]
junior_max_length=c_maxlength(train_junior_sentence,junior_max_length)
junior_max_length=c_maxlength(test_junior_sentence,junior_max_length)
c_w2idx(train_junior_sentence, w2idx)
c_w2idx(test_junior_sentence, w2idx)
# In[句子转换并且填充]
def sen2idx(sen_list,sen_target,w2idx,max_length):
    word=[]
    word_target=[]
    word_mask=[]
    txt=[]
    target=[]
    mask=[]
    for i in range(len(sen_list)):
        for j in range(len(sen_list[i])):
            word.append(w2idx[sen_list[i][j]])
            word_target.append(target_to_num[sen_target[i][j]])
            word_mask+=[1]
        if(len(word)<max_length):
            word+=[0]*(max_length-len(word))
            word_target+=[0]*(max_length-len(word_target))
            word_mask+=[0]*(max_length-len(word_mask))
        txt.append(word)
        target.append(word_target)
        mask.append(word_mask)
        word=[]
        word_target=[]
        word_mask=[]
    return txt,target,mask
# In[数据集准备]
class MyData(Dataset):
    
    def __init__(self,sentence,sen_target,w2idx,max_length):
        super().__init__()
        self.sentence,self.sen_target,self.mask=sen2idx(sentence,sen_target,w2idx,max_length)
        self.sentence=torch.LongTensor(self.sentence)
        self.sen_target=torch.LongTensor(self.sen_target)
        self.mask=torch.BoolTensor(self.mask)           
    def __getitem__(self,idx):
        return self.sentence[idx],self.sen_target[idx],self.mask[idx]
    
    def __len__(self):
        return len(self.sentence)
# In[BILSTM-CRF]
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, target_size, embedding_dim, num_layers, hidden_dim, batch_size):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size =target_size
        self.num_layers=num_layers
        self.batch_size = batch_size
        self.word_embeds = nn.Embedding(vocab_size,embedding_dim)
        self.conv=nn.Conv2d(64,64,kernel_size=(3,50),padding=(1,0))
        self.maxpool=nn.MaxPool2d(kernel_size=(3,22),stride=1
                                  ,padding=(1,0))
        self.cnn2embedding = nn.Linear(embedding_dim+30, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers =self.num_layers,
                            bidirectional = True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim*2, self.tagset_size)
        self.crf = CRF(self.tagset_size)
 
   
    def forward(self, sentence,target=None,mask=None):
        self.batch_size = sentence.shape[0]
        embeds = self.word_embeds(sentence)
        x=self.conv(embeds)
        x=F.relu(x)
        x=self.maxpool(x)
        embeds = torch.cat((embeds,x),dim=2)
        embeds = self.cnn2embedding(embeds)
        self.hidden = (torch.randn(2*self.num_layers,self.batch_size,self.hidden_dim,device=sentence.device)
                       ,torch.randn(2*self.num_layers,self.batch_size,self.hidden_dim,device=sentence.device))
        lstm_out, self.hidden = self.lstm(embeds, self.hidden) 
        lstm_feats = self.hidden2tag(lstm_out)
        if target is not None:
            loss = (-1)*self.crf(lstm_feats,target,mask)
            return loss
        else:
            result=self.crf.viterbi_decode(lstm_feats, mask)
            return result
# In[训练集和测试集]
train_data=MyData(train_junior_sentence,train_junior_target, w2idx, junior_max_length)
test_data=MyData(test_junior_sentence,test_junior_target, w2idx, junior_max_length)
# In[对训练集和测试集进行打包]
train_loader=DataLoader(train_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
test_loader=DataLoader(test_data,batch_size=64,shuffle=True,num_workers=0,drop_last=True)
# In[创建模型和优化器]
model = BiLSTM_CRF(len(w2idx)+1, 3, embedding_dim, num_layers,hidden_dim,batch_size)
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)
# In[模型训练]
# for i in range(300):
#     for data in train_loader:
#         sentence,target,mask=data
#         model.zero_grad()
#         if torch.cuda.is_available():
#             sentence=sentence.cuda()
#             target=target.cuda()
#             mask=mask.cuda()
#         loss=model(sentence,target,mask)
#         optimizer.zero_grad()
#         loss.sum().backward()
#         optimizer.step()
#     if i % 10 == 0 :
#         print(loss.mean().item())
# # In[保存模型]
# torch.save(model,'D:/毕业设计/src/junior_CNN_model.pth')
# In[导出模型]
model=torch.load('D:/毕业设计/src/junior_CNN_model.pth')
# In[测试集计算正确率]
right=0
total=0
for data in test_loader:
    sentence,target,mask=data
    if torch.cuda.is_available():
        sentence=sentence.cuda()
        target=target.cuda()
        mask=mask.cuda()
    x=sentence
    y=mask
    result=model(sentence=sentence,mask=mask)
    for i in range(len(result)):
        for j in range(len(result[i])):
            total+=1
            if(target[i][j].item()==result[i][j]):
                right+=1
print("正确率为",right/total)
# In[]
y_true=[]
y_pred=[]
for data in test_loader:
    sentence,target,mask=data
    if torch.cuda.is_available():
        sentence=sentence.cuda()
        target=target.cuda()
        mask=mask.cuda()
    x=sentence
    y=mask
    result=model(sentence=sentence,mask=mask)
    for i in range(len(result)):
        for j in range(len(result[i])):
            y_true.append(target[i][j].item())
            y_pred.append(result[i][j])
# In[]
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
recall_score = recall_score(y_true, y_pred, average=None)
f1 = f1_score(y_true, y_pred, average=None)
precision = precision_score(y_true, y_pred, average=None)
# In[]
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["O", "B-MAT", "I-MAT"], cmap=plt.cm.Reds, colorbar=True)
plt.title("Confusion Matrix")    
    
    