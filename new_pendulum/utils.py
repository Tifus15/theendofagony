import numpy as np
import torch
import dgl
import matplotlib.pyplot as plt
import random 
from dgl.dataloading import GraphDataLoader

def make_train_test_loader(data,split,batch_size,time_size,nodes,src,dst):
    x_temp = data[:,:,:,:-1]
    H_temp = data[:,:,:,-1]
    xs,hs = make_snapshots(x_temp.float(),H_temp.float().unsqueeze(-1),time_size)
    print(xs[0].shape)
    border = int(split*len(xs))
    c = list(zip(xs, hs))
    random.shuffle(c)
    xs, hs = zip(*c)
    train = xs[0:border]
    test = xs[border:]

    h_train = hs[0:border]
    h_test = hs[border:]
    if nodes == 3:
        train_snapshots = create_pend3dof_graph_snapshots(train,h_train,src,dst)
        test_snapshots = create_pend3dof_graph_snapshots(test,h_test,src,dst)
    elif nodes ==4:
        train_snapshots = create_pend4dof_graph_snapshots(train,h_train,src,dst)
        test_snapshots = create_pend4dof_graph_snapshots(test,h_test,src,dst)
    trainset = GraphDataLoader(train_snapshots,batch_size=batch_size,drop_last=True,shuffle=True)
    it = iter(trainset)
    g = next(it)
    #model.change_graph(g)
    N_train=len(trainset)
    print("TRAIN BATCHES : {}".format(N_train))
    testset = GraphDataLoader(test_snapshots,batch_size=batch_size,drop_last=True,shuffle=True)
    it = iter(testset)
    gt = next(it)
    
    N_test=len(testset)
    print("TEST BATCHES : {}".format(N_test))
    return trainset, testset,N_train,N_test
    
    
    
    ts = t[0:TIME_SIZE]
    
    

def dst_list(nodes,start=0):

    base = list(np.arange(start,start+nodes))
    out=[]
    for i in range(nodes):
        out = out + base
    return out

def src_list(nodes,start=0):
    out=[]
    for i in range(nodes):
        out = out +list(np.zeros((nodes),dtype=int)+start+i)
    return out

def del_loops(src,dst):
    for i, (val1,val2) in enumerate(zip(src,dst)):
        
        if val1 == val2:
            #print("{} == {}".format(val1,val2))
            #print("{} popped".format(i))
            src.pop(i)
            dst.pop(i)
    return src, dst

def make_graph_no_loops(nodes,start):
    src = src_list(nodes,start)
    dst = dst_list(nodes,start)
    return del_loops(src,dst)

"""
def load_dataset(file,hfile):
    data = torch.load(file)
    H = torch.load(hfile)
    return data, H
"""
def make_simple_snapshots(data,H):
    xlist=[]
    xnext_list=[]
    Hlist=[]
    N = data.shape[1]
    print(N)
    T = data.shape[0]
    for i in range(N):
        for j in range(T-1):
            temp = data[j,i,:,:]
            tempH = H[j,i,:]
            tempx = data[j+1,i,:,:]
            #print("simp x:{}".format(temp.shape))
            #print("simp nx:{}".format(tempx.shape))
            xlist.append(temp)
            Hlist.append(tempH)
            xnext_list.append(tempx)
    return xlist, Hlist



def make_snapshots(data,H,timesize):
    xlist=[]
    Hlist=[]
    
    
    N = data.shape[1]
    print(N)
    T = data.shape[0]
    for i in range(N):
        for j in range(T-timesize-1):
            temp = data[j:timesize+j,i,:,:]
           #print("temp {}".format(temp.shape))
            #print(i)
            tempH = H[j:timesize+j,i,:,:]
            xlist.append(temp)
            Hlist.append(tempH)
    return xlist, Hlist

def transform_dgl(src,dst,snaps,hs):
    gs = []
    for snap,h in zip(snaps,hs):
        g = dgl.graph((src,dst))
        g.ndata["x"] = snap[:,:,0:6].transpose(0,1)
        g.ndata["dx"] = snap[:,:,6:].transpose(0,1)
        g.ndata["H"] = h[:,:,0:1].transpose(0,1)
        gs.append(g)
    return gs
def get_d_dx_H(sample):
    gs = dgl.unbatch(sample)
    H = []
    for g in gs:
        h_raw = g.ndata["H"].transpose(0,1)
        H.append(h_raw[:,0,0:1])
    H_out = torch.cat((H),dim=-1)
    x_out = sample.ndata["x"].transpose(0,1)
    dx_out = sample.ndata["dx"].transpose(0,1)
    
    return x_out, dx_out, H_out


    
    
'''
def dataset_loader(first=["4_x.pt","4_dx.pt","4_h.pt"],second=["5_x.pt","5_dx.pt","5_h.pt"]):
    x1 = torch.load(first[0])
    dx1= torch.load(first[1])
    H1 = torch.load(first[2])
    x2 = torch.load(second[0])
    dx2= torch.load(second[1])
    H2 = torch.load(second[2])
    
    
    dxx1 = torch.cat((x1,dx1),dim=-1)
    dxx2 = torch.cat((x2,dx2),dim=-1)
    
    return dxx1, H1 ,dxx2, H2
'''
def minmax(dataset):
    T = dataset.shape[0]
    B = dataset.shape[1]
    maxim=torch.max(dataset.flatten())
    minim=torch.min(dataset.flatten())
    #print(maxim.shape)
    return (dataset - minim)/(maxim-minim), maxim, minim


def inv_minmax(dataset,min_key,max_key):
    return (dataset*(max_key-min_key))+min_key


def minimax_test(dataset):
    d , maxim, minim = minmax(dataset)
    rec = inv_minmax(d,minim,maxim)
    return torch.mean(rec.flatten() - dataset.flatten())

def loss_reader(str):
    if str == "MSE":
        return torch.nn.MSELoss()
    elif str == "HUB":
        return torch.nn.HuberLoss()
    else:
        return torch.nn.MSELoss()
    

def RKroll_for_learning(model,x0,t):
    def evaluate_model(model,x):
        h_pred = model(x)
        H_l = torch.split(h_pred,1,dim=1)
        dHdx = torch.autograd.grad(H_l,x,retain_graph=True, create_graph=True)[0] 
        dqdp_s = torch.split(dHdx,1,dim=-1)
        dx_pred = torch.cat((dqdp_s[1],-dqdp_s[0]),dim=-1)
        return dx_pred
    out_l = []
    out_l.append(x0.unsqueeze(0))
    #print(out_l[0].shape)
    for i in range(1,len(t)):
        dt=t[i]-t[i-1]
        K1 = evaluate_model(model,out_l[i-1].squeeze())
        K2 = evaluate_model(model,out_l[i-1].squeeze()+dt*K1/2)
        K3 = evaluate_model(model,out_l[i-1].squeeze()+dt*K2/2)
        K4 = evaluate_model(model,out_l[i-1].squeeze()+dt*K3)
        rk4=out_l[i-1].squeeze()+dt*(K1+2*K2+2*K3+K4)/6
        out_l.append(rk4.unsqueeze(0))
        #print(out_l[i].shape)
    
    return torch.cat((out_l),dim=0)

def Euler_for_learning(model,x0,t):
    def evaluate_model(model,x):
        h_pred = model(x)
        H_l = torch.split(h_pred,1,dim=1)
        dHdx = torch.autograd.grad(H_l,x,retain_graph=True, create_graph=True)[0] 
        dqdp_s = torch.split(dHdx,1,dim=-1)
        dx_pred = torch.cat((dqdp_s[1],-dqdp_s[0]),dim=-1)
        return dx_pred
    out_l = []
    out_l.append(x0.unsqueeze(0))
    #print(out_l[0].shape)
    for i in range(1,len(t)):
        dt=t[i]-t[i-1]
        K1 = evaluate_model(model,out_l[i-1].squeeze())
        rk4=out_l[i-1].squeeze()+dt*K1
        out_l.append(rk4.unsqueeze(0))
        #print(out_l[i].shape)
    
    return torch.cat((out_l),dim=0)






def angle_transformer(data):
    x = torch.cos(data)
    y = torch.sin(data)
    out = torch.atan2(y,x)
    return out
    
    
def check_angle_transformer():
    x = -100 + torch.rand(128,100,2)*200
    test = torch.cos(x)
    y = angle_transformer(x)
    print(torch.max(y.flatten()))# pi
    print(torch.min(y.flatten()))# -pi
    y_test = torch.cos(y)
    print(torch.mean(torch.abs(y_test-test).flatten())) # minimal mistake
    #

def create_pend1dof_graph_snapshots(xsnaps,hsnaps,src,dst):
    graphs=[]
    for xsnap, hsnap in zip(xsnaps,hsnaps):
        #print(snap.shape) 
        g = dgl.graph((src,dst))
        g.ndata["x"] = xsnap[:,:,0:2].transpose(0,1) 
        g.ndata["dx"] = xsnap[:,:,2:4].transpose(0,1)
        g.ndata["h"] =  hsnap.transpose(0,1)
        graphs.append(g)
    return graphs

def visualize_loss(title, loss_container):
    if loss_container.shape[0]==2:
        t = torch.linspace(0,loss_container.shape[1],loss_container.shape[1])
        fig = plt.figure()
        plt.title("{} logy".format(title))
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.semilogy(t,loss_container[0,:].detgach().numpy(),c="r")
        plt.semilogy(t,loss_container[1,:].detach().numpy(),c="b")
        plt.legend(["train","test"])
        fig.savefig("loss_"+title+".png")
    else:
        t = torch.linspace(0,loss_container.shape[1],loss_container.shape[1])
        fig,ax = plt.subplots(1,2,sharey=True)
        ax[0].set_title("{} loss".format(title))
        ax[0].set_xlabel("epochs")
        ax[0].set_ylabel("loss")
        ax[0].semilogy(t,loss_container[0,:].detach().numpy(),c="r")
        ax[0].semilogy(t,loss_container[2,:].detach().numpy(),c="b")
        ax[0].legend(["train","test"])
        ax[1].set_title("{} hamiltonian loss".format(title))
        ax[1].set_xlabel("epochs")
        ax[1].set_ylabel("loss")
        ax[1].semilogy(t,loss_container[1,:].detach().numpy(),c="r")
        ax[1].semilogy(t,loss_container[3,:].detach().numpy(),c="b")
        ax[1].legend(["train ham","test ham"]) 
        fig.savefig("loss_"+title+".png")
        
def create_pend2dof_graph_snapshots(xsnaps,hsnaps,src,dst):
    graphs=[]
    nodes = 2
    for xsnap,hsnap in zip(xsnaps,hsnaps):
        #print(snap.shape)
        tempx=torch.zeros(2,xsnap.shape[0],2) 
        tempx[0,:,0] = xsnap[:,0,0]
        tempx[1,:,0] = xsnap[:,0,1]
        tempx[0,:,1] = xsnap[:,0,2]
        tempx[1,:,1] = xsnap[:,0,3]
        tempdx=torch.zeros(2,xsnap.shape[0],2) 
        tempdx[0,:,0] = xsnap[:,0,4]
        tempdx[1,:,0] = xsnap[:,0,5]
        tempdx[0,:,1] = xsnap[:,0,6]
        tempdx[1,:,1] = xsnap[:,0,7]
        tempH=torch.zeros(2,xsnap.shape[0],1)
        tempH[0,:,0] = hsnap[:,0,0]
        tempH[1,:,0] = hsnap[:,0,0]
        g = dgl.graph((src,dst))
        g.ndata["x"] = tempx
        g.ndata["dx"] = tempdx
        g.ndata["h"] =  tempH
        graphs.append(g)
    return graphs

def correct_ham_data(g):
    gs = dgl.unbatch(g)
    l = []
    for s in gs:
        temp = s.ndata["h"] 
        #print(temp)
        l.append(temp[0,:,:])
    return torch.cat((l),dim=-1)


def create_pend3dof_graph_snapshots(xsnaps,hsnaps,src,dst):
    graphs=[]
    nodes = 3
    for xsnap,hsnap in zip(xsnaps,hsnaps):
        #print(snap.shape)
        tempx=torch.zeros(3,xsnap.shape[0],2) 
        tempx[0,:,0] = xsnap[:,0,0]
        tempx[1,:,0] = xsnap[:,0,1]
        tempx[2,:,0] = xsnap[:,0,2]
        tempx[0,:,1] = xsnap[:,0,3]
        tempx[1,:,1] = xsnap[:,0,4]
        tempx[2,:,1] = xsnap[:,0,5]
        tempdx=torch.zeros(3,xsnap.shape[0],2) 
        tempdx[0,:,0] = xsnap[:,0,6]
        tempdx[1,:,0] = xsnap[:,0,7]
        tempdx[2,:,0] = xsnap[:,0,8]
        tempdx[0,:,1] = xsnap[:,0,9]
        tempdx[1,:,1] = xsnap[:,0,10]
        tempdx[2,:,1] = xsnap[:,0,11]
        tempH=torch.zeros(3,xsnap.shape[0],1)
        tempH[0,:,0] = hsnap[:,0,0]
        tempH[1,:,0] = hsnap[:,0,0]
        tempH[2,:,0] = hsnap[:,0,0]
        
        g = dgl.graph((src,dst))
        g.ndata["x"] = tempx
        g.ndata["dx"] = tempdx
        g.ndata["h"] =  tempH
        graphs.append(g)
    return graphs 

def load_model(model, loc_w):
     # we do not specify ``weights``, i.e. create untrained model
    model.load_state_dict(torch.load(loc_w))
    return model

def create_pend4dof_graph_snapshots(xsnaps,hsnaps,src,dst):
    graphs=[]
    nodes = 4
    for xsnap, hsnap in zip(xsnaps,hsnaps):
        #print(snap.shape)
        tempx=torch.zeros(4,xsnap.shape[0],2) 
        tempx[0,:,0] = xsnap[:,0,0]
        tempx[1,:,0] = xsnap[:,0,1]
        tempx[2,:,0] = xsnap[:,0,2]
        tempx[3,:,0] = xsnap[:,0,3]
        tempx[0,:,1] = xsnap[:,0,4]
        tempx[1,:,1] = xsnap[:,0,5]
        tempx[2,:,1] = xsnap[:,0,6]
        tempx[3,:,1] = xsnap[:,0,7]
        tempdx=torch.zeros(4,xsnap.shape[0],2) 
        tempdx[0,:,0] = xsnap[:,0,8]
        tempdx[1,:,0] = xsnap[:,0,9]
        tempdx[2,:,0] = xsnap[:,0,10]
        tempdx[3,:,0] = xsnap[:,0,11]
        tempdx[0,:,1] = xsnap[:,0,12]
        tempdx[1,:,1] = xsnap[:,0,13]
        tempdx[2,:,1] = xsnap[:,0,14]
        tempdx[3,:,1] = xsnap[:,0,15]
        tempH=torch.zeros(4,xsnap.shape[0],1)
        tempH[0,:,0] = hsnap[:,0,0]
        tempH[1,:,0] = hsnap[:,0,0]
        tempH[2,:,0] = hsnap[:,0,0]
        tempH[3,:,0] = hsnap[:,0,0]
        
        g = dgl.graph((src,dst))
        g.ndata["x"] = tempx
        g.ndata["dx"] = tempdx
        g.ndata["h"] =  tempH
        graphs.append(g)
    return graphs


def RKroll_for_learning(model,x0,t):
    def evaluate_model(model,x):
        h_pred = model(x)
        H_l = torch.split(h_pred,1,dim=1)
        dHdx = torch.autograd.grad(H_l,x,retain_graph=True, create_graph=True)[0] 
        dqdp_s = torch.split(dHdx,1,dim=-1)
        dx_pred = torch.cat((dqdp_s[1],-dqdp_s[0]),dim=-1)
        return dx_pred,h_pred
    out_l = []
    out_dl= []
    out_h = []
    out_l.append(x0.unsqueeze(0))
    dx , h = evaluate_model(model,x0)
    out_dl.append(dx.unsqueeze(0))
    out_h.append(h.unsqueeze(0))
    #print(out_l[0].shape)
    for i in range(1,len(t)):
        dt=t[i]-t[i-1]
        K1,h_pred = evaluate_model(model,out_l[i-1].squeeze())
        K2, _ = evaluate_model(model,out_l[i-1].squeeze()+dt*K1/2)
        K3, _ = evaluate_model(model,out_l[i-1].squeeze()+dt*K2/2)
        K4, _ = evaluate_model(model,out_l[i-1].squeeze()+dt*K3)
        rk4=out_l[i-1].squeeze()+dt*(K1+2*K2+2*K3+K4)/6
        #print("RK4")
        #print(h_pred.shape)
        #print(rk4.shape)
        #print("###")
        out_l.append(rk4.unsqueeze(0))
        out_dl.append(K1.unsqueeze(0))
        out_h.append(h_pred.unsqueeze(0))
        #print(out_l[i].shape)
    
    return torch.cat((out_l),dim=0), torch.cat((out_dl),dim=0),torch.cat((out_h),dim=0).squeeze()
