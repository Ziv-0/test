# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 13:46:10 2020

@author: Administrator
"""

# -*- coding: utf-8 -*-
#hooker 机制，让所有训练好后触发hooker
"""
Created on Mon Oct  5 18:51:23 2020

@author: Administrator
"""
from visdom import Visdom
import torch
import torchvision
import threading
import time




class Fcnmodel(torch.nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Fcnmodel, self).__init__()
        self.layer1 = torch.nn.Linear(in_dim, n_hidden_1)
        self.layer2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = torch.nn.Linear(n_hidden_2, out_dim)
 
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class Worker():
    def __init__(self):
        pass
 
class Server():
    def __init__(self):
        pass
    
def worker_train(ni,worker,nepoch,batch_size,sleeptime):
    for nth_epoch in range(nepoch):
        if sleeptime != 0:
            time.sleep(sleeptime)
        for i,sampledata in enumerate(worker.dataloader):
                labels = sampledata[1].to(device)
                pilimage = sampledata[0].to(device)
                a = pilimage.reshape(batch_size,784)
                predict = worker.model(a)
                # labels = torch.zeros(batch_size, class_num).scatter_(1, labels.unsqueeze(1), 1)
                lossfunc = torch.nn.CrossEntropyLoss()
                loss = lossfunc(predict,labels)
                # print(loss)
                worker.optim.zero_grad()
                loss.backward()
                worker.optim.step()
        num_right = torch.IntTensor([0]).to(device)
        with torch.no_grad():
            for i,sampledata in enumerate(test_dataloader):
                labels = sampledata[1].to(device)
                pilimage = sampledata[0].to(device)
                a = pilimage.reshape(batch_size,784)
                predict = worker.model(a)
                pre = predict.max(1)[1]
                num_right += sum(torch.eq(pre,labels))
        acc = num_right.float()/(i*batch_size)
        print('worker: %d epoch:%d acc:%f' % (ni,nth_epoch,acc))
        
def para_aggre(worker_list,para_key):
    temp_tensor = worker_list[0].model.state_dict()[para_key]
    for i in range(1,len(worker_list)):
        temp_tensor = torch.add(temp_tensor,worker_list[i].model.state_dict()[para_key])
    return temp_tensor/(len(worker_list))

if __name__ == '__main__':
    #可视化环境
    viz = Visdom()
    viz.line([0.], [0], win='train', opts=dict(title='server_acc'))
    
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
    # device = 'cpu'
    bigdataset = torchvision.datasets.MNIST('./',train = True,transform =torchvision.transforms.ToTensor(),target_transform=None,download=False)
    class_num = 10
    batch_size = 100
    worker_N = 5
    sleeptime_list = [0,0,0,0,0]
    num_dataloader_workers = 3
    smalldataset_size = int(len(bigdataset )/(worker_N+1))
    dataset_list = torch.utils.data.random_split(bigdataset,[smalldataset_size for i in range(worker_N+1)])
    test_dataloader = torch.utils.data.DataLoader(dataset_list[-1],batch_size= batch_size,shuffle=True,num_workers=num_dataloader_workers)
    
    #server 初始化
    server = Server()
    server.model = Fcnmodel(784,10,10,10).to(device)
    #worker 初始化
    worker_list =[]
    for i in range(worker_N):
        worker_list.append(Worker())
        worker_list[i].dataloader = torch.utils.data.DataLoader(dataset_list[i],batch_size= batch_size,shuffle=True,num_workers=num_dataloader_workers)
        worker_list[i].model = Fcnmodel(784,10,10,10).to(device)
        worker_list[i].optim = torch.optim.SGD(worker_list[i].model.parameters(), lr = 0.01, momentum=0)
    #开始训练
    niter = 10000
    nepoch = 2
    for i_iter in range(niter):
    #train woker i
        threads = []
        for k in range(len(worker_list)):
            t = threading.Thread(target= worker_train,args=(k, worker_list[k], nepoch, batch_size,sleeptime_list[k]))
            t.setDaemon(True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    
    #server参数整合
        with torch.no_grad():
            for para_key in  server.model.state_dict().keys():
                eval('server.model'+'.'+para_key).set_(para_aggre(worker_list,para_key))    
                
    #server准确率计算
        num_right = torch.IntTensor([0]).to(device)
        with torch.no_grad():
            for i,sampledata in enumerate(test_dataloader):
                labels = sampledata[1].to(device)
                pilimage = sampledata[0].to(device)
                a = pilimage.reshape(batch_size,784)
                predict =server.model(a)
                pre = predict.max(1)[1]
                num_right += sum(torch.eq(pre,labels))
        acc = num_right.float()/(i*batch_size)
        viz.line([acc], [i_iter], win='train', update='append')
        print('niter: %d server test acc:%f' % (i_iter,acc))
        
    #server参数分发
        for agi in range(len(worker_list)):
            with torch.no_grad():
                for para_key in  server.model.state_dict().keys():
                    current_para = eval('worker_list[agi].model.'+para_key)
                    current_para.data.copy_(eval('server.model.'+para_key+'.data'))
        pass