import torch
import dgl
from utils import *
import random
import yaml
from ghnn_model import *
from graph_model import GNN_HNN
import wandb
from tqdm import tqdm

from dgl.dataloading import GraphDataLoader
#from torch_symplectic_adjoint import odeint
from torchdiffeq import odeint
import os

def new_full(configs):
    MODEL = configs["model"] #["GCN","GAT"]
    SOB = configs["sob"] # sobolev - gradients training
    s_alpha = configs["sob_a"]
    alpha = configs["a"]
    OPTI = configs["opti"] # ["adamW","RMS","SGD"]
    LOSS = configs["loss"] # ["MSE","MAE","Huber"]
    REG = configs["reg"] #["lasso","ridge","none"]
    ACT_FUNC = configs["acts"] # activations - don't touch this
    BIAS = configs["bias"]
    WANDB = configs["wandb"]

    MODEL_SIZE = configs["modelsize"]
    #DATASETSIZE = 512
    #SINGLE = configs["single"]
    S= configs["samples"]
    EPOCHS = configs["epochs"]
    BATCH_SIZE = configs["batchsize"]
    TIME_SIZE = configs["timesize"]
    LR = configs["lr"]
    SPLIT = configs["split"]
    DATASET = "3dof pendelum"
    print(EPOCHS)
    NOLOOPS = configs["noloops"]
    REG = "ridge"
    t = torch.linspace(0,1.27,128)[0:TIME_SIZE]
    EVAL = configs["eval"]
    
 
    data3 = torch.load("pend_3.pt").requires_grad_(False)
    print(data3.shape)
    data4 = torch.load("pend_4.pt").requires_grad_(False)
    print(data4.shape)
    
    src3 = src_list(3)
    dst3 = dst_list(3)
    src4 = src_list(4)
    dst4 = dst_list(4)
    if NOLOOPS:
        src3,dst3 = del_loops(src3,dst3)
        src4,dst4 = del_loops(src4,dst4)
 
    graph3 = dgl.graph((src3,dst3))
    graph4 = dgl.graph((src4,dst4))
    dim = 2
    #print(H[:,0,0])
    
    data3[:,:,:,0:3] = angle_transformer(data3[:,:,:,0:3])
    data4[:,:,:,0:4] = angle_transformer(data4[:,:,:,0:4])
    
    eval3 = data3[:,S:S+EVAL,:,:]
    eval4 = data4[:,S:S+EVAL,:,:]
    
    data3 = data3[:,:S,:,:]
    print(data3.shape)
    data4 = data4[:,:S,:,:]
    print(data4.shape)
    
    trainset3,testset3,N_train3,N_test3 =make_train_test_loader(data3,SPLIT,BATCH_SIZE,TIME_SIZE,3,src3,dst3)
    trainset4,testset4,N_train4,N_test4 =make_train_test_loader(data4,SPLIT,BATCH_SIZE,TIME_SIZE,4,src4,dst4)

    
    del data3, data4
    
    ts = t[0:TIME_SIZE]
    model31 = rollout_GNN_GRU(graph3,2,128,8,["tanh"," "],bias=BIAS,type = MODEL,dropout=0.65)
    model41 = rollout_GNN_GRU(graph4,2,128,8,["tanh"," "],bias=BIAS,type = MODEL,dropout=0.65)
    print(model31)
    print(model41)
    model32 = GNN_HNN(graph3,2,128,8,["tanh",""])
    model42 = GNN_HNN(graph4,2,128,8,["tanh",""])
    print(model32)
    print(model42)
    
    if OPTI=="RMS":
        opti31 = torch.optim.RMSprop(model31.parameters(),lr=LR)
        opti32 = torch.optim.RMSprop(model32.parameters(),lr=LR)
        opti41 = torch.optim.RMSprop(model41.parameters(),lr=LR)
        opti42 = torch.optim.RMSprop(model42.parameters(),lr=LR)
    if OPTI=="SGD":
        opti31 = torch.optim.SGD(model31.parameters(),lr=LR)
        opti32= torch.optim.SGD(model32.parameters(),lr=LR)
        opti41 = torch.optim.SGD(model41.parameters(),lr=LR)
        opti42 = torch.optim.SGD(model42.parameters(),lr=LR)
    if OPTI == "adamW":
        opti31 = torch.optim.AdamW(model31.parameters(),lr=LR)
        opti32= torch.optim.AdamW(model32.parameters(),lr=LR)
        opti41 = torch.optim.AdamW(model41.parameters(),lr=LR)
        opti42 = torch.optim.AdamW(model42.parameters(),lr=LR)
    

    if LOSS == "MSE":
        lossfn = nn.MSELoss()
    if LOSS == "MAE":
        lossfn = nn.L1Loss()
    if LOSS == "Huber":
        lossfn = nn.HuberLoss()
    
    metrics={"train_sum1":0,  "train_roll1":0, "train_vec1" :0, "train_h1" :0,
             "test_sum1":0,  "test_roll1":0, "test_vec1" :0, "test_h1" :0,
             "train_sum2":0,  "train_roll2":0, "train_vec2" :0, "train_h2" :0,
             "test_sum2":0,  "test_roll2":0, "test_vec2" :0, "test_h2" :0,
             "train_sum3":0,  "train_roll3":0, "train_vec3" :0, "train_h3" :0,
             "test_sum3":0,  "test_roll3":0, "test_vec3" :0, "test_h3" :0,
             "train_sum4":0,  "train_roll4":0, "train_vec4" :0, "train_h4" :0,
             "test_sum4":0,  "test_roll4":0, "test_vec4" :0, "test_h4" :0,}
        

    container = torch.zeros(32,EPOCHS) #["train loss, train grad loss, test loss, test grad loss"]

    
    #wandb.watch(model,log='all')
    
    for epoch in tqdm(range(EPOCHS)):
        
        
        it3 = iter(trainset3)
        it4 = iter(trainset4)
        model31.train()
        model32.train()
        model41.train()
        model42.train()
        for i in tqdm(range(N_train3)):
            opti31.zero_grad()
            opti32.zero_grad()
            opti41.zero_grad()
            opti42.zero_grad()
            
            loss31_train=0
            loss32_train=0
            loss41_train=0
            loss42_train=0
            loss31_roll=0
            loss32_roll=0
            loss41_roll=0
            loss42_roll=0
            loss31_vec=0
            loss32_vec=0
            loss41_vec=0
            loss42_vec=0
            loss31_h=0
            loss32_h=0
            loss41_h=0
            loss42_h=0
            
            
            train_sample3 = next(it3)
            train_sample4 = next(it4)
            #print(i)
            #print(sample3)
            #print(sample4)
            x_tr3 = train_sample3.ndata["x"].transpose(0,1)
            dx_tr3 = train_sample3.ndata["dx"].transpose(0,1)
            h_tr3 = correct_ham_data(train_sample3)
            
            x_tr4 = train_sample4.ndata["x"].transpose(0,1)
            dx_tr4 = train_sample4.ndata["dx"].transpose(0,1)
            h_tr4 = correct_ham_data(train_sample4)
        
            #print(x_tr4.shape)
            #print(x_tr3.shape)
            
            x03 = x_tr3[0,:,:].requires_grad_()
            x04 = x_tr4[0,:,:].requires_grad_()
            model31.change_graph(train_sample3)
            model41.change_graph(train_sample4)
            x_pred31, dx_pred31, h_pred31 = model31(ts,x03)
            x_pred41, dx_pred41, h_pred41 = model41(ts,x04)
            #print(x_tr_flat.shape)
            #h_pred = model(x_tr_flat)
            #print(h_pred.shape)
            #print(h_tr.reshape(-1,1).shape)
            loss31_h = lossfn(h_pred31.flatten(),h_tr3.flatten())
            loss41_h = lossfn(h_pred41.flatten(),h_tr4.flatten())
            loss31_vec = lossfn(dx_pred31[:,:,0],dx_tr3[:,:,0])+lossfn(dx_pred31[:,:,1],dx_tr3[:,:,1])
            loss41_vec = lossfn(dx_pred41[:,:,0],dx_tr4[:,:,0])+lossfn(dx_pred41[:,:,1],dx_tr4[:,:,1])
            loss31_roll = lossfn(x_pred31[:,:,0],x_tr3[:,:,0])+lossfn(x_pred31[:,:,1],x_tr3[:,:,1])
            loss41_roll = lossfn(x_pred41[:,:,0],x_tr4[:,:,0])+lossfn(x_pred41[:,:,1],x_tr4[:,:,1])
            
            
            model32.change_graph(train_sample3)
            model42.change_graph(train_sample4)
            x_pred32,dx_pred32,h_pred32 = RKroll_for_learning(model32,x03,ts)
            x_pred42,dx_pred42,h_pred42 = RKroll_for_learning(model42,x04,ts)
            
            loss32_h = lossfn(h_pred32.flatten(),h_tr3.flatten())
            loss42_h = lossfn(h_pred42.flatten(),h_tr4.flatten())
            loss32_vec = lossfn(dx_pred32[:,:,0],dx_tr3[:,:,0])+lossfn(dx_pred32[:,:,1],dx_tr3[:,:,1])
            loss42_vec = lossfn(dx_pred42[:,:,0],dx_tr4[:,:,0])+lossfn(dx_pred42[:,:,1],dx_tr4[:,:,1])
            loss32_roll = lossfn(x_pred32[:,:,0],x_tr3[:,:,0])+lossfn(x_pred32[:,:,1],x_tr3[:,:,1])
            loss42_roll = lossfn(x_pred42[:,:,0],x_tr4[:,:,0])+lossfn(x_pred42[:,:,1],x_tr4[:,:,1])
            
            loss31_train = (s_alpha[0]*loss31_roll + s_alpha[1]*loss31_vec + s_alpha[2]*loss31_h)
            loss32_train = (s_alpha[0]*loss32_roll + s_alpha[1]*loss32_vec + s_alpha[2]*loss32_h)
            loss41_train = (s_alpha[0]*loss41_roll + s_alpha[1]*loss41_vec + s_alpha[2]*loss41_h)
            loss42_train = (s_alpha[0]*loss42_roll + s_alpha[1]*loss42_vec + s_alpha[2]*loss42_h)
            
            container[0,epoch] += loss31_train.item()
            container[1,epoch] += loss31_roll.item()
            container[2,epoch] += loss31_vec.item()
            container[3,epoch] += loss31_h.item()
            container[4,epoch] += loss41_train.item()
            container[5,epoch] += loss41_roll.item()
            container[6,epoch] += loss41_vec.item()
            container[7,epoch] += loss41_h.item()
            container[8,epoch] += loss32_train.item()
            container[9,epoch] += loss32_roll.item()
            container[10,epoch] += loss32_vec.item()
            container[11,epoch] += loss32_h.item()
            container[12,epoch] += loss42_train.item()
            container[13,epoch] += loss42_roll.item()
            container[14,epoch] += loss42_vec.item()
            container[15,epoch] += loss42_h.item()
            
            loss31_train.backward()
            loss32_train.backward()
            loss41_train.backward()
            loss42_train.backward()
            
            opti31.step()
            opti32.step()
            opti41.step()
            opti42.step()
        itt3 = iter(testset3)
        itt4 = iter(testset4)
        model31.eval()
        model32.eval()
        model41.eval()
        model42.eval()
        for i in tqdm(range(N_test3)):
           
            loss31_test=0
            loss32_test=0
            loss41_test=0
            loss42_test=0
            loss31_roll_t=0
            loss32_roll_t=0
            loss41_roll_t=0
            loss42_roll_t=0
            loss31_vec_t=0
            loss32_vec_t=0
            loss41_vec_t=0
            loss42_vec_t=0
            loss31_h_t=0
            loss32_h_t=0
            loss41_h_t=0
            loss42_h_t=0
            
            
            test_sample3 = next(itt3)
            test_sample4 = next(itt4)
            #print(i)
            #print(sample3)
            #print(sample4)
            x_tr3 = test_sample3.ndata["x"].transpose(0,1)
            dx_tr3 = test_sample3.ndata["dx"].transpose(0,1)
            h_tr3 = correct_ham_data(test_sample3)
            
            x_tr4 = test_sample4.ndata["x"].transpose(0,1)
            dx_tr4 = test_sample4.ndata["dx"].transpose(0,1)
            h_tr4 = correct_ham_data(test_sample4)
        
            #print(x_tr4.shape)
            #print(x_tr3.shape)
            
            x03 = x_tr3[0,:,:].requires_grad_()
            x04 = x_tr4[0,:,:].requires_grad_()
            model31.change_graph(test_sample3)
            model41.change_graph(test_sample4)
            x_pred31, dx_pred31, h_pred31 = model31(ts,x03)
            x_pred41, dx_pred41, h_pred41 = model41(ts,x04)
            #print(x_tr_flat.shape)
            #h_pred = model(x_tr_flat)
            #print(h_pred.shape)
            #print(h_tr.reshape(-1,1).shape)
            loss31_h_t = lossfn(h_pred31.flatten(),h_tr3.flatten())
            loss41_h_t = lossfn(h_pred41.flatten(),h_tr4.flatten())
            loss31_vec_t = lossfn(dx_pred31[:,:,0],dx_tr3[:,:,0])+lossfn(dx_pred31[:,:,1],dx_tr3[:,:,1])
            loss41_vec_t = lossfn(dx_pred41[:,:,0],dx_tr4[:,:,0])+lossfn(dx_pred41[:,:,1],dx_tr4[:,:,1])
            loss31_roll_t = lossfn(x_pred31[:,:,0],x_tr3[:,:,0])+lossfn(x_pred31[:,:,1],x_tr3[:,:,1])
            loss41_roll_t = lossfn(x_pred41[:,:,0],x_tr4[:,:,0])+lossfn(x_pred41[:,:,1],x_tr4[:,:,1])
            
            
            model32.change_graph(test_sample3)
            model42.change_graph(test_sample4)
            x_pred32,dx_pred32,h_pred32 = RKroll_for_learning(model32,x03,ts)
            x_pred42,dx_pred42,h_pred42 = RKroll_for_learning(model42,x04,ts)
            
            loss32_h_t = lossfn(h_pred32.flatten(),h_tr3.flatten())
            loss42_h_t = lossfn(h_pred42.flatten(),h_tr4.flatten())
            loss32_vec_t = lossfn(dx_pred32[:,:,0],dx_tr3[:,:,0])+lossfn(dx_pred32[:,:,1],dx_tr3[:,:,1])
            loss42_vec_t = lossfn(dx_pred42[:,:,0],dx_tr4[:,:,0])+lossfn(dx_pred42[:,:,1],dx_tr4[:,:,1])
            loss32_roll_t = lossfn(x_pred32[:,:,0],x_tr3[:,:,0])+lossfn(x_pred32[:,:,1],x_tr3[:,:,1])
            loss42_roll_t = lossfn(x_pred42[:,:,0],x_tr4[:,:,0])+lossfn(x_pred42[:,:,1],x_tr4[:,:,1])
            
            loss31_test = (s_alpha[0]*loss31_roll + s_alpha[1]*loss31_vec + s_alpha[2]*loss31_h)
            loss32_test = (s_alpha[0]*loss32_roll + s_alpha[1]*loss32_vec + s_alpha[2]*loss32_h)
            loss41_test = (s_alpha[0]*loss41_roll + s_alpha[1]*loss41_vec + s_alpha[2]*loss41_h)
            loss42_test = (s_alpha[0]*loss42_roll + s_alpha[1]*loss42_vec + s_alpha[2]*loss42_h)
            
            container[16,epoch] += loss31_test.item()
            container[17,epoch] += loss31_roll_t.item()
            container[18,epoch] += loss31_vec_t.item()
            container[19,epoch] += loss31_h_t.item()
            container[20,epoch] += loss41_test.item()
            container[21,epoch] += loss41_roll_t.item()
            container[22,epoch] += loss41_vec_t.item()
            container[23,epoch] += loss41_h_t.item()
            container[24,epoch] += loss32_test.item()
            container[25,epoch] += loss32_roll_t.item()
            container[26,epoch] += loss32_vec_t.item()
            container[27,epoch] += loss32_h_t.item()
            container[28,epoch] += loss42_test.item()
            container[29,epoch] += loss42_roll_t.item()
            container[30,epoch] += loss42_vec_t.item()
            container[31,epoch] += loss42_h_t.item() 
        container[0:16,epoch]/=N_train3    
        container[16:,epoch]/=N_test3
        if WANDB:
            metrics["train_sum1"]= container[0,epoch]
            metrics["train_roll1"]= container[1,epoch]
            metrics["train_vec1"]= container[2,epoch]
            metrics["train_h1"]= container[3,epoch]
            metrics["train_sum2"]= container[4,epoch]
            metrics["train_roll2"]= container[5,epoch]
            metrics["train_vec2"]= container[6,epoch]
            metrics["train_h2"]= container[7,epoch]
            metrics["train_sum3"]= container[8,epoch]
            metrics["train_roll3"]= container[9,epoch]
            metrics["train_vec3"]= container[10,epoch]
            metrics["train_h3"]= container[11,epoch]
            metrics["train_sum4"]= container[12,epoch]
            metrics["train_roll4"]= container[13,epoch]
            metrics["train_vec4"]= container[14,epoch]
            metrics["train_h4"]= container[15,epoch]
            metrics["test_sum1"]= container[16,epoch]
            metrics["test_roll1"]= container[17,epoch]
            metrics["test_vec1"]= container[18,epoch]
            metrics["test_h1"]= container[19,epoch]
            metrics["test_sum2"]= container[20,epoch]
            metrics["test_roll2"]= container[21,epoch]
            metrics["test_vec2"]= container[22,epoch]
            metrics["test_h2"]= container[23,epoch]
            metrics["test_sum3"]= container[24,epoch]
            metrics["test_roll3"]= container[25,epoch]
            metrics["test_vec3"]= container[26,epoch]
            metrics["test_h3"]= container[27,epoch]
            metrics["test_sum4"]= container[28,epoch]
            metrics["test_roll4"]= container[29,epoch]
            metrics["test_vec4"]= container[30,epoch]
            metrics["test_h4"]= container[31,epoch]
            wandb.log(metrics)
        print("GRUGHNN")
        print("train node3:")
        print("sum: {}  roll: {}  vec: {}  h:{}".format(container[0,epoch],
                                                        container[1,epoch],
                                                        container[2,epoch],
                                                        container[3,epoch]))
        print("test node3")  
        print("sum: {}  roll: {}  vec: {}  h:{}".format(container[16,epoch],
                                                        container[17,epoch],
                                                        container[18,epoch],
                                                        container[19,epoch]))
        print("train node4:")
        print("sum: {}  roll: {}  vec: {}  h:{}".format(container[4,epoch],
                                                        container[5,epoch],
                                                        container[6,epoch],
                                                        container[7,epoch]))
        print("test node4")  
        print("sum: {}  roll: {}  vec: {}  h:{}".format(container[20,epoch],
                                                        container[21,epoch],
                                                        container[22,epoch],
                                                        container[23,epoch]))   
        print("GHNN")
        print("train node3:")
        print("sum: {}  roll: {}  vec: {}  h:{}".format(container[8,epoch],
                                                        container[9,epoch],
                                                        container[10,epoch],
                                                        container[11,epoch]))
        print("test node3")  
        print("sum: {}  roll: {}  vec: {}  h:{}".format(container[24,epoch],
                                                        container[25,epoch],
                                                        container[26,epoch],
                                                        container[27,epoch]))
        print("train node4:")
        print("sum: {}  roll: {}  vec: {}  h:{}".format(container[12,epoch],
                                                        container[13,epoch],
                                                        container[14,epoch],
                                                        container[15,epoch]))
        print("test node4")  
        print("sum: {}  roll: {}  vec: {}  h:{}".format(container[28,epoch],
                                                        container[29,epoch],
                                                        container[30,epoch],
                                                        container[31,epoch])) 
            
    torch.save(container,"results/losses.pt")
    torch.save(eval3,"results/eval3.pt")  
    torch.save(eval4,"results/eval4.pt")
    torch.save(model31,"results/model31.pt") 
    torch.save(model32,"results/model32.pt") 
    torch.save(model41,"results/model41.pt") 
    torch.save(model42,"results/model42.pt") 
            
            







def full_server43(configs,dic_base):
    print("begin 4dof")
    train4dof(configs,dic_base)
    print("end 4dof")
    print("begin 3dof")
    train3dof(configs,dic_base)
    print("end 3dof")
   

def full_server34(configs,dic_base):
    """
    print("begin 1dof")
    train1dof(configs)
    print("end 1dof")
    print("begin 2dof")
    
    train2dof(configs)
    print("end 2dof")
    """
    print("begin 3dof")
    train3dof(configs,dic_base)
    print("end 3dof")
    print("begin 4dof")
    train4dof(configs,dic_base)
    print("end 4dof")
    
def train1dof(configs):
    

    MODEL = configs["model"] #["GCN","GAT"]
    SOB = configs["sob"] # sobolev - gradients training
    s_alpha = configs["sob_a"]
    alpha = configs["a"]
    OPTI = configs["opti"] # ["adamW","RMS","SGD"]
    LOSS = configs["loss"] # ["MSE","MAE","Huber"]
    REG = configs["reg"] #["lasso","ridge","none"]
    ACT_FUNC = configs["acts"] # activations - don't touch this
    NOLOOPS = configs["noloops"]
    WANDB = True
    BIAS = configs["bias"]
    S = configs["samples"]
    MODEL_SIZE = configs["modelsize"]
    #DATASETSIZE = 512
    #SINGLE = configs["single"]

    EPOCHS = configs["epochs"]
    BATCH_SIZE = configs["batchsize"]
    TIME_SIZE = configs["timesize"]
    LR = configs["lr"]
    SPLIT = configs["split"]
    DATASET = "1dof pendelum"
    print(EPOCHS)
    DOF = 1
    t = torch.linspace(0,1.27,128)[0:TIME_SIZE]
    
    
 
    data = torch.load("traj_1dof.pt").requires_grad_(False)
    """
    H = data[:,:,:,-1]
    x = data[:,:,:,0:2]
    x[:,:,:,0] = angle_transformer(x[:,:,:,0]) # to have it between -pi and pi
    dx = data[:,:,:,2:4]


    print(data.shape)
    print(H.shape)
    print(x.shape)
    print(dx.shape)
    """
    src = src_list(1)
    dst = dst_list(1)
    if NOLOOPS and DOF != 1 :
        src,dst = del_loops(src,dst)
        
 
    graph = dgl.graph((src,dst))
    dim = 2
    #print(H[:,0,0])
    
    
    model = GNN_maker_HNN(graph,2,128,6,["tanh",""],type=MODEL,bias=BIAS)
    print(model)
    
    
    data[:,:,:,0] = angle_transformer(data[:,:,:,0])
    
    
    #num = random.randint(0,dataset.shape[1]-1)
    eval = data[:,-1,:,:].unsqueeze(1)
    H = eval[:,:,-1]
    data = data[:,:S,:,:]
    print(data.shape)

    x_temp = data[:,:,:,0:4]
    H_temp = data[:,:,:,-1]
    xs,hs = make_snapshots(x_temp.float(),H_temp.float().unsqueeze(-1),TIME_SIZE) # just 128 to keep everything in 2^i
    
    print(xs[0].shape)
    border = int(SPLIT*len(xs))
    c = list(zip(xs, hs))
    random.shuffle(c)
    xs, hs = zip(*c)
    train = xs[0:border]
    test = xs[border:]

    h_train = hs[0:border]
    h_test = hs[border:]
    

    train_snapshots = create_pend1dof_graph_snapshots(train,h_train,src,dst)
    test_snapshots = create_pend1dof_graph_snapshots(test,h_test,src,dst)
    #graph_snapshots = make_graph_snapshots(snapshots,nodes=6,feats=4)

    #dgl_snapshots = convert2dgl_snapshots(snapshots,src,dst)
    

 
    ts = t[0:TIME_SIZE]

   
    print(model)
    if OPTI=="RMS":
        opti = torch.optim.RMSprop(model.parameters(),lr=LR)
    if OPTI=="SGD":
        opti = torch.optim.SGD(model.parameters(),lr=LR)
    if OPTI == "adamW":
        opti = torch.optim.AdamW(model.parameters(),lr=LR)

    if LOSS == "MSE":
        lossfn = nn.MSELoss()
    if LOSS == "MAE":
        lossfn = nn.L1Loss()
    if LOSS == "Huber":
        lossfn = nn.HuberLoss()

    
    trainset = GraphDataLoader(train_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(trainset)
    g = next(it)
    #model.change_graph(g)
    N_train=len(trainset)
    print("TRAIN BATCHES : {}".format(N_train))
    testset = GraphDataLoader(test_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(testset)
    gt = next(it)
    
    N_test=len(testset)
    print("TEST BATCHES : {}".format(N_test))
    gs=[]
    for i in range(TIME_SIZE*BATCH_SIZE):
        if DOF != 1:
            src, dst = make_graph_no_loops(1,0)
        else:
            src = src_list(1)
            dst = dst_list(1)
        gtemp = dgl.graph((src,dst))
        #print(g.num_nodes())
        gs.append(gtemp)
    #print(len(gs))
    #print(g.num_nodes())
    roll_g = dgl.batch(gs)
    
    
    metrics={"train_loss_d1":0,  "train_H_d1":0, "test_loss_d1" :0, "test_H_d1" :0}
        

    container = torch.zeros(4,EPOCHS) #["train loss, train grad loss, test loss, test grad loss"]

    
    wandb.watch(model,log='all')
    
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        
        print("TRAIN")
        
        for train_sample in tqdm(trainset):
            #print(train_sample.num_nodes())
        
            loss=0
            lossH=0
            lossroll=0
            opti.zero_grad()
            #model.set_graph(train_sample)
            x_tr = train_sample.ndata["x"].transpose(0,1)
            dx_tr = train_sample.ndata["dx"].transpose(0,1)
            h_tr = train_sample.ndata["h"].transpose(0,1)
            x0 = x_tr[0,:,:]
            #print(x_tr.shape)
            #print(dx_tr.shape)
            #print(h_tr.shape)
            model.change_graph(roll_g)
            #print(roll_g)
            x_tr = x_tr.requires_grad_()
            x_tr_flat = x_tr.reshape(-1,2)
            #print(x_tr_flat.shape)
            h_pred = model(x_tr_flat)
            lossH = lossfn(h_pred.flatten(),h_tr.flatten())
            """
            if REG == "ridge":
                loss += alpha[0] * sum(p.square().sum() for p in model.parameters())
            if REG == "lasso":
                loss += alpha * sum(p.abs().sum() for p in model.parameters())
            """
            x0 = x_tr[0,:,:].requires_grad_()
            model.change_graph(train_sample)
            x_pred = Euler_for_learning(model,x0,ts)
            lossroll = lossfn(x_pred[:,:,0],x_tr[:,:,0])+lossfn(x_pred[:,:,1],x_tr[:,:,1])

            #print("after h roll")
            loss += s_alpha[0]* lossroll
            loss += s_alpha[1]* lossH
            container[1,epoch]+=lossH.item()
            container[0,epoch] += loss.item()
            
            loss.backward()
            opti.step()
        container[0:2,epoch]/=N_train
        print("TEST")
        model.eval()
        
        
        for test_sample in tqdm(testset):
            losst=0
            lossHt=0
            lossROLLt=0
            model.change_graph(test_sample)
            x_ts = test_sample.ndata["x"].transpose(0,1)
            dx_ts = test_sample.ndata["dx"].transpose(0,1)
            h_ts = test_sample.ndata["h"].transpose(0,1)
            model.change_graph(roll_g)
            x_ts = x_ts.requires_grad_()
            x_ts_flat = x_ts.reshape(-1,2)
            h_pred = model(x_ts_flat)
            lossHt = lossfn(h_pred.flatten(),h_ts.flatten())
            x0 = x_ts[0,:,:].requires_grad_()
            model.change_graph(test_sample)
            x_pred = Euler_for_learning(model,x0,ts)
            lossROLLt = lossfn(x_pred[:,:,0],x_ts[:,:,0])+lossfn(x_pred[:,:,1],x_ts[:,:,1])
        
            losst+=s_alpha[1] * lossHt
            losst+=s_alpha[0] * lossROLLt
                
           
            container[2,epoch]+=losst.item()
            container[3,epoch] += lossHt.item()
        container[2:4,epoch]/=N_test
    
        metrics["train_loss_d1"] = container[0,epoch]
        metrics["test_loss_d1"] = container[2,epoch]
        metrics["train_H_d1"] = container[1,epoch]
        metrics["test_H_d1"] = container[3,epoch]
        wandb.log(metrics)
            #wandb.log_artifact(model)
       
        print("Epoch: {}\nLOSS: train: {:.6f}   ham: {:.6f} |   test: {:.6f}  ham: {:.6f}".format(epoch+1,container[0,epoch],container[1,epoch],container[2,epoch],container[3,epoch]))
        dict={}
        for namew , param in model.named_parameters():
            dict[namew+"_grad"] = torch.mean(param.grad).item()
        print(dict)
   
    
    visualize_loss("loss of 1dof pendelum",container)
    torch.save(model.state_dict(),"server_1dof.pth")
    #torch.save(model,"whole_model_dof1.pt")
    
def train2dof(configs):
    S = configs["samples"]
    MODEL = configs["model"] #["GCN","GAT"]
    SOB = configs["sob"] # sobolev - gradients training
    s_alpha = configs["sob_a"]
    alpha = configs["a"]
    OPTI = configs["opti"] # ["adamW","RMS","SGD"]
    LOSS = configs["loss"] # ["MSE","MAE","Huber"]
    REG = configs["reg"] #["lasso","ridge","none"]
    ACT_FUNC = configs["acts"] # activations - don't touch this
    NOLOOPS = configs["noloops"]
    WANDB = True
    BIAS = configs["bias"]

    MODEL_SIZE = configs["modelsize"]
    #DATASETSIZE = 512
 

    EPOCHS = configs["epochs"]
    BATCH_SIZE = configs["batchsize"]
    TIME_SIZE = configs["timesize"]
    LR = configs["lr"]
    SPLIT = configs["split"]
    DATASET = "2dof pendelum"
    print(EPOCHS)
    
    t = torch.linspace(0,1.27,128)[0:TIME_SIZE]
    
    
 
    data = torch.load("traj_2dof.pt").requires_grad_(False)
    """
    H = data[:,:,:,-1]
    x = data[:,:,:,0:2]
    x[:,:,:,0] = angle_transformer(x[:,:,:,0]) # to have it between -pi and pi
    dx = data[:,:,:,2:4]


    print(data.shape)
    print(H.shape)
    print(x.shape)
    print(dx.shape)
    """
    src = src_list(2)
    dst = dst_list(2)
    if NOLOOPS:
        src,dst = del_loops(src,dst)
 
    graph = dgl.graph((src,dst))
    dim = 2
    #print(H[:,0,0])
    data[:,:,:,0:2] = angle_transformer(data[:,:,:,0:2])

    
        #num = random.randint(0,dataset.shape[1]-1)
    eval = data[:,-1,:,:].unsqueeze(1)
    H = eval[:,:,-1]
    data = data[:,:S,:,:]
    print(data.shape)
    
    x_temp = data[:,:,:,:-1]
    H_temp = data[:,:,:,-1]

    
    xs, hs = make_snapshots(x_temp.float(),H_temp.float().unsqueeze(-1),TIME_SIZE) # just 128 to keep everything in 2^i
  
    
    print(xs[0].shape)
    border = int(SPLIT*len(xs))
    c = list(zip(xs, hs))
    random.shuffle(c)
    xs, hs = zip(*c)
    train = xs[0:border]
    test = xs[border:]

    h_train = hs[0:border]
    h_test = hs[border:]
    
    train_snapshots = create_pend2dof_graph_snapshots(train,h_train,src,dst)
    test_snapshots = create_pend2dof_graph_snapshots(test,h_test,src,dst)
    
    
    
    ts = t[0:TIME_SIZE]

    #half = int(dim/6) 
    model = GNN_maker_HNN(graph,2,128,6,["tanh",""],type=MODEL,bias =BIAS)
    if os.path.isfile("server_1dof.pth"):
        print("loading prevoius model")
        model = load_model(model,"server_1dof.pth")
    

    print(model)
    if OPTI=="RMS":
        opti = torch.optim.RMSprop(model.parameters(),lr=LR)
    if OPTI=="SGD":
        opti = torch.optim.SGD(model.parameters(),lr=LR)
    if OPTI == "adamW":
        opti = torch.optim.AdamW(model.parameters(),lr=LR)

    if LOSS == "MSE":
        lossfn = nn.MSELoss()
    if LOSS == "MAE":
        lossfn = nn.L1Loss()
    if LOSS == "Huber":
        lossfn = nn.HuberLoss()

   
    
 

    trainset = GraphDataLoader(train_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(trainset)
    g = next(it)
    #model.change_graph(g)
    N_train=len(trainset)
    print("TRAIN BATCHES : {}".format(N_train))
    testset = GraphDataLoader(test_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(testset)
    gt = next(it)
    
    N_test=len(testset)
    print("TEST BATCHES : {}".format(N_test))
    gs=[]
    for i in range(TIME_SIZE*BATCH_SIZE):
        if NOLOOPS:
            src, dst = make_graph_no_loops(2,0)
        else:
            src = src_list(2)
            dst = dst_list(2)
        gtemp = dgl.graph((src,dst))
        #print(g.num_nodes())
        gs.append(gtemp)
    #print(len(gs))
    #print(g.num_nodes())
    roll_g = dgl.batch(gs)
    
    
    metrics={"train_loss_d2":0,  "train_H_d2":0, "test_loss_d2" :0, "test_H_d2" :0}
        

    container = torch.zeros(4,EPOCHS) #["train loss, train grad loss, test loss, test grad loss"]

    
    wandb.watch(model,log='all')
    
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        
        print("TRAIN")
        
        for train_sample in tqdm(trainset):
            #print(train_sample.num_nodes())
        
            loss=0
            lossH=0
            lossroll=0
            opti.zero_grad()
            #model.set_graph(train_sample)
            x_tr = train_sample.ndata["x"].transpose(0,1)
            dx_tr = train_sample.ndata["dx"].transpose(0,1)
            #h_tr = train_sample.ndata["h"].transpose(0,1)
            h_tr = correct_ham_data(train_sample)
    
            x0 = x_tr[0,:,:]
            #print(x_tr.shape)
            #print(dx_tr.shape)
            #print(h_tr.shape)
            model.change_graph(roll_g)
            #print(roll_g)
            x_tr = x_tr.requires_grad_()
            x_tr_flat = x_tr.reshape(-1,2)
            
            #print(x_tr_flat.shape)
            h_pred = model(x_tr_flat)
            #print(h_pred.shape)
            #print(h_tr.reshape(-1,1).shape)
            lossH = lossfn(h_pred.flatten(),h_tr.flatten())
            """
            if REG == "ridge":
                loss += alpha[0] * sum(p.square().sum() for p in model.parameters())
            if REG == "lasso":
                loss += alpha * sum(p.abs().sum() for p in model.parameters())
            """
            x0 = x_tr[0,:,:].requires_grad_()
            model.change_graph(train_sample)
            x_pred = Euler_for_learning(model,x0,ts)
            lossroll = lossfn(x_pred[:,:,0],x_tr[:,:,0])+lossfn(x_pred[:,:,1],x_tr[:,:,1])

            #print("after h roll")
            loss += s_alpha[0]* lossroll
            loss += s_alpha[1]* lossH
            container[1,epoch]+=lossH.item()
            container[0,epoch] += loss.item()
            
            loss.backward()
            opti.step()
        container[0:2,epoch]/=N_train
        print("TEST")
        model.eval()
        
        
        for test_sample in tqdm(testset):
            losst=0
            lossHt=0
            lossROLLt=0
            model.change_graph(test_sample)
            x_ts = test_sample.ndata["x"].transpose(0,1)
            dx_ts = test_sample.ndata["dx"].transpose(0,1)
            h_ts = h_tr = correct_ham_data(test_sample)
            model.change_graph(roll_g)
            x_ts = x_ts.requires_grad_()
            x_ts_flat = x_ts.reshape(-1,2)
            h_pred = model(x_ts_flat)
            lossHt = lossfn(h_pred.flatten(),h_ts.flatten())
            x0 = x_ts[0,:,:].requires_grad_()
            model.change_graph(test_sample)
            x_pred = Euler_for_learning(model,x0,ts)
            lossROLLt = lossfn(x_pred[:,:,0],x_ts[:,:,0])+lossfn(x_pred[:,:,1],x_ts[:,:,1])
        
            losst+=s_alpha[1] * lossHt
            losst+=s_alpha[0] * lossROLLt
                
           
            container[2,epoch]+=losst.item()
            container[3,epoch] += lossHt.item()
        container[2:4,epoch]/=N_test
    
        metrics["train_loss_d2"] = container[0,epoch]
        metrics["test_loss_d2"] = container[2,epoch]
        metrics["train_H_d2"] = container[1,epoch]
        metrics["test_H_d2"] = container[3,epoch]
        wandb.log(metrics)
            #wandb.log_artifact(model)
        dict={}
        print("Epoch: {}\nLOSS: train: {:.6f}   ham: {:.6f} |   test: {:.6f}  ham: {:.6f}".format(epoch+1,container[0,epoch],container[1,epoch],container[2,epoch],container[3,epoch]))
        for namew , param in model.named_parameters():
            dict[namew+"_grad"] = torch.mean(param.grad).item()
        print(dict)
   
   
    
    visualize_loss("loss of 2dof pendelum",container)
    torch.save(model.state_dict(),"server_2dof.pth")
    #torch.save(model,"whole_model_dof2.pt")
    
def train3dof(configs,dic_base):
    
    MODEL = configs["model"] #["GCN","GAT"]
    SOB = configs["sob"] # sobolev - gradients training
    s_alpha = configs["sob_a"]
    alpha = configs["a"]
    OPTI = configs["opti"] # ["adamW","RMS","SGD"]
    LOSS = configs["loss"] # ["MSE","MAE","Huber"]
    REG = configs["reg"] #["lasso","ridge","none"]
    ACT_FUNC = configs["acts"] # activations - don't touch this
    BIAS = configs["bias"]
    WANDB = True

    MODEL_SIZE = configs["modelsize"]
    #DATASETSIZE = 512
    #SINGLE = configs["single"]
    S= configs["samples"]
    EPOCHS = configs["epochs"]
    BATCH_SIZE = configs["batchsize"]
    TIME_SIZE = configs["timesize"]
    LR = configs["lr"]
    SPLIT = configs["split"]
    DATASET = "3dof pendelum"
    print(EPOCHS)
    NOLOOPS = configs["noloops"]
    REG = "ridge"
    t = torch.linspace(0,1.27,128)[0:TIME_SIZE]
    
    
 
    data = torch.load("pend_3.pt").requires_grad_(False)
    """
    H = data[:,:,:,-1]
    x = data[:,:,:,0:2]
    x[:,:,:,0] = angle_transformer(x[:,:,:,0]) # to have it between -pi and pi
    dx = data[:,:,:,2:4]


    print(data.shape)
    print(H.shape)
    print(x.shape)
    print(dx.shape)
    """
    src = src_list(3)
    dst = dst_list(3)
    if NOLOOPS:
        src,dst = del_loops(src,dst)
 
    graph = dgl.graph((src,dst))
    dim = 2
    #print(H[:,0,0])
    data[:,:,:,0:3] = angle_transformer(data[:,:,:,0:3]) # to be sure

    
        #num = random.randint(0,dataset.shape[1]-1)
    eval = data[:,-1,:,:].unsqueeze(1)
    H = data[:,-1,:,-1]
    data = data[:,:S,:,:]
    print(data.shape)

    
    x_temp = data[:,:,:,:-1]
    H_temp = data[:,:,:,-1]
    xs,hs = make_snapshots(x_temp.float(),H_temp.float().unsqueeze(-1),TIME_SIZE) # just 128 to keep everything in 2^i
    
    
    print(xs[0].shape)
    border = int(SPLIT*len(xs))
    c = list(zip(xs, hs))
    random.shuffle(c)
    xs, hs = zip(*c)
    train = xs[0:border]
    test = xs[border:]

    h_train = hs[0:border]
    h_test = hs[border:]
    train_snapshots = create_pend3dof_graph_snapshots(train,h_train,src,dst)
    test_snapshots = create_pend3dof_graph_snapshots(test,h_test,src,dst)
    
    ts = t[0:TIME_SIZE]

    #half = int(dim/6) 
    model = rollout_GNN_GRU(graph,2,128,8,["tanh"," "],bias=BIAS,type = MODEL,dropout=0.65)
    #model = GNN_maker_HNN(graph,2,128,6,["tanh",""],type=MODEL,bias=BIAS)
    #if os.path.isfile("server_2dof.pth"):
        #print("loading prevoius model")
        #model = load_model(model,"server_2dof.pth")
    
    dic = "res_4dof/"
    if os.path.isfile(dic_base+"/"+dic+"server_4dof.pth"):
        print("loading prevoius model")
        model.train()
        model = load_model(model,dic_base+"/"+dic+"server_4dof.pth")
        
        

    print(model)
    if OPTI=="RMS":
        opti = torch.optim.RMSprop(model.parameters(),lr=LR)
    if OPTI=="SGD":
        opti = torch.optim.SGD(model.parameters(),lr=LR)
    if OPTI == "adamW":
        opti = torch.optim.AdamW(model.parameters(),lr=LR)
    if os.path.isfile(dic_base+"/"+dic+"server_3dof.pth"):
        opti.load_state_dict(torch.load(dic_base+"/"+dic+"server_opti.pth"))

    if LOSS == "MSE":
        lossfn = nn.MSELoss()
    if LOSS == "MAE":
        lossfn = nn.L1Loss()
    if LOSS == "Huber":
        lossfn = nn.HuberLoss()

    trainset = GraphDataLoader(train_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(trainset)
    g = next(it)
    #model.change_graph(g)
    N_train=len(trainset)
    print("TRAIN BATCHES : {}".format(N_train))
    testset = GraphDataLoader(test_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(testset)
    gt = next(it)
    
    N_test=len(testset)
    print("TEST BATCHES : {}".format(N_test))
   
    
    metrics={"train_sum":0,  "train_roll":0, "train_vec" :0, "train_h" :0,
             "test_sum":0,  "test_roll":0, "test_vec" :0, "test_h" :0}
        

    container = torch.zeros(8,EPOCHS) #["train loss, train grad loss, test loss, test grad loss"]

    
    #wandb.watch(model,log='all')
    
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        
        print("TRAIN")
        
        for train_sample in tqdm(trainset):
            #print(train_sample.num_nodes())
        
            loss=0
            lossH=0
            lossvec=0
            lossroll=0
            reg = 0
            opti.zero_grad()
            #model.set_graph(train_sample)
            x_tr = train_sample.ndata["x"].transpose(0,1)
            dx_tr = train_sample.ndata["dx"].transpose(0,1)
            #h_tr = train_sample.ndata["h"].transpose(0,1)
            h_tr = correct_ham_data(train_sample)
    
            x0 = x_tr[0,:,:].requires_grad_()
            #print(x_tr.shape)
            #print(dx_tr.shape)
            #print(h_tr.shape)
            #model.change_graph(roll_g)
            #print(roll_g)
            #x_tr_flat = x_tr.reshape(-1,2)
            model.change_graph(train_sample)
            x_pred, dx_pred, h_pred = model(ts,x0)
            #print(x_tr_flat.shape)
            #h_pred = model(x_tr_flat)
            #print(h_pred.shape)
            #print(h_tr.reshape(-1,1).shape)
            lossH = lossfn(h_pred.flatten(),h_tr.flatten())
            lossvec = lossfn(dx_pred[:,:,0],dx_tr[:,:,0])+lossfn(dx_pred[:,:,1],dx_tr[:,:,1])
            lossroll = lossfn(x_pred[:,:,0],x_tr[:,:,0])+lossfn(x_pred[:,:,1],x_tr[:,:,1])
            #if REG == "ridge":
            #    reg= sum(p.square().sum() for p in model.parameters())
            #if REG == "lasso":
            #    reg= sum(p.abs().sum() for p in model.parameters())
            #print("after h roll")
            loss += s_alpha[0]* lossroll
            loss += s_alpha[1]* lossH
            loss += s_alpha[2]*lossvec
            loss += 0.01 * reg
            container[0,epoch] += loss.item()
            container[1,epoch]+=lossroll.item()
            container[2,epoch]+=lossvec.item()
            container[3,epoch]+=lossH.item()
            
            loss.backward()
            opti.step()
        container[0:4,epoch]/=N_train
        print("TEST")
        model.eval()
        
        
        for test_sample in tqdm(testset):
            losst=0
            lossHt=0
            lossvect=0
            lossROLLt=0
            model.change_graph(test_sample)
            x_ts = test_sample.ndata["x"].transpose(0,1)
            dx_ts = test_sample.ndata["dx"].transpose(0,1)
            h_ts = correct_ham_data(test_sample)
            #print(h_ts)
            #print(h_ts.shape)
            
            model.change_graph(test_sample)
            x_ts = x_ts.requires_grad_()
            x0 = x_ts[0,:,:]
            #x_ts_flat = x_ts.reshape(-1,2)
            #h_pred = model(x_ts_flat)
            x_pred,dx_pred,h_pred =model(ts,x0.requires_grad_()) 
            lossHt = lossfn(h_pred.flatten(),h_ts.flatten())

            lossROLLt = lossfn(x_pred[:,:,0],x_ts[:,:,0])+lossfn(x_pred[:,:,1],x_ts[:,:,1])
            lossvect = lossfn(dx_pred[:,:,0],dx_ts[:,:,0])+lossfn(dx_pred[:,:,1],dx_ts[:,:,1])
        
            
            losst+=s_alpha[0] * lossROLLt
            losst+=s_alpha[1] * lossHt
            losst+=s_alpha[2] * lossvect 
                
           
            container[4,epoch]+=losst.item()
            container[5,epoch]+=lossROLLt.item()
            container[6,epoch]+=lossvect.item()
            container[7,epoch] += lossHt.item()
        container[4:8,epoch]/=N_test
    
        metrics["train_sum"] = container[0,epoch]
        metrics["train_roll"] = container[1,epoch]
        metrics["train_vec"] = container[2,epoch]
        metrics["train_h"] = container[3,epoch]
        metrics["test_sum"] = container[4,epoch]
        metrics["test_roll"] = container[5,epoch]
        metrics["test_vec"] = container[6,epoch]
        metrics["test_h"] = container[7,epoch]
        #wandb.log(metrics)
            #wandb.log_artifact(model)
        print("\nEpoch: {}\nLOSS: train: {:.6f} roll: {:.6f} vec: {:.6f}  ham: {:.6f} \n".format(epoch+1,
                                                                                            container[0,epoch],
                                                                                            container[1,epoch],
                                                                                            container[2,epoch],
                                                                                            container[3,epoch],)+
                                "      test: {:.6f} roll: {:.6f} vec: {:.6f}  ham: {:.6f}".format(container[4,epoch],
                                                                                            container[5,epoch],
                                                                                            container[6,epoch],
                                                                                            container[7,epoch]))
        
   
   
    model.train()
    dic = "res_3dof/"
    #visualize_loss("loss of 3dof pendelum",container)
    torch.save(model.state_dict(),dic_base+"/"+dic+"server_3dof.pth")
    torch.save(opti.state_dict(),dic_base+"/"+dic+"server_opti.pth")
    torch.save(container,dic_base+"/"+dic+"losses.pt")
    model.eval()
    torch.save(model,dic_base+"/"+dic+"model.pt")
    torch.save(eval,dic_base+"/"+dic+"eval.pt")
    torch.save(H,dic_base+"/"+dic+"eval_H.pt")
    #torch.save(model,"whole_model_dof3.pt")
    
def train4dof(configs,dic_base):

    MODEL = configs["model"] #["GCN","GAT"]
    SOB = configs["sob"] # sobolev - gradients training
    s_alpha = configs["sob_a"]
    alpha = configs["a"]
    OPTI = configs["opti"] # ["adamW","RMS","SGD"]
    LOSS = configs["loss"] # ["MSE","MAE","Huber"]
    REG = configs["reg"] #["lasso","ridge","none"]
    ACT_FUNC = configs["acts"] # activations - don't touch this

    WANDB = True

    MODEL_SIZE = configs["modelsize"]
    #DATASETSIZE = 512
    #SINGLE = configs["single"]
    S = configs["samples"]
    EPOCHS = configs["epochs"]
    BATCH_SIZE = configs["batchsize"]
    TIME_SIZE = configs["timesize"]
    LR = configs["lr"]
    SPLIT = configs["split"]
    DATASET = "4dof pendelum"
    BIAS = configs["bias"]
    print(EPOCHS)
    
    t = torch.linspace(0,1.27,128)[0:TIME_SIZE]
    
    
    NOLOOPS = configs["noloops"]
    data = torch.load("pend_4.pt").requires_grad_(False)
    """
    H = data[:,:,:,-1]
    x = data[:,:,:,0:2]
    x[:,:,:,0] = angle_transformer(x[:,:,:,0]) # to have it between -pi and pi
    dx = data[:,:,:,2:4]


    print(data.shape)
    print(H.shape)
    print(x.shape)
    print(dx.shape)
    """
    src = src_list(4)
    dst = dst_list(4)
    if NOLOOPS:
        src,dst = del_loops(src,dst)
 
    graph = dgl.graph((src,dst))
    dim = 2
    #print(H[:,0,0])
    data[:,:,:,0:4] = angle_transformer(data[:,:,:,0:4])

    
        #num = random.randint(0,dataset.shape[1]-1)
    eval = data[:,-1,:,:].unsqueeze(1)
    H = data[:,-1,:,-1]
    data = data[:,:S,:,:]
    print(data.shape)
    x_temp = data[:,:,:,:-1]
    H_temp = data[:,:,:,-1]
    
    xs,hs = make_snapshots(x_temp.float(),H_temp.float().unsqueeze(-1),TIME_SIZE)
    
    print(xs[0].shape)
    border = int(SPLIT*len(xs))
    c = list(zip(xs, hs))
    random.shuffle(c)
    xs, hs = zip(*c)
    train = xs[0:border]
    test = xs[border:]

    h_train = hs[0:border]
    h_test = hs[border:]
    train_snapshots = create_pend4dof_graph_snapshots(train,h_train,src,dst)
    test_snapshots = create_pend4dof_graph_snapshots(test,h_test,src,dst)
    
    ts = t[0:TIME_SIZE]
    dic = "res_3dof/"
    #half = int(dim/6) 
    model = rollout_GNN_GRU(graph,2,128,8,["tanh"," "],bias=BIAS,type = MODEL,dropout=0.65)
    if os.path.isfile(dic_base+"/"+dic+"server_3dof.pth"):
        print("loading prevoius model")
        model.train()
        model = load_model(model,dic_base+"/"+dic+"server_3dof.pth")
        
        

    print(model)
    if OPTI=="RMS":
        opti = torch.optim.RMSprop(model.parameters(),lr=LR)
    if OPTI=="SGD":
        opti = torch.optim.SGD(model.parameters(),lr=LR)
    if OPTI == "adamW":
        opti = torch.optim.AdamW(model.parameters(),lr=LR)
    if os.path.isfile(dic+"server_3dof.pth"):
        opti.load_state_dict(torch.load(dic_base+"/"+dic+"server_opti.pth"))
        
    if LOSS == "MSE":
        lossfn = nn.MSELoss()
    if LOSS == "MAE":
        lossfn = nn.L1Loss()
    if LOSS == "Huber":
        lossfn = nn.HuberLoss()

    trainset = GraphDataLoader(train_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(trainset)
    g = next(it)
    #model.change_graph(g)
    N_train=len(trainset)
    print("TRAIN BATCHES : {}".format(N_train))
    testset = GraphDataLoader(test_snapshots,batch_size=BATCH_SIZE,drop_last=True,shuffle=True)
    it = iter(testset)
    gt = next(it)
    
    N_test=len(testset)
    print("TEST BATCHES : {}".format(N_test))
    gs=[]
    for i in range(TIME_SIZE*BATCH_SIZE):
        if NOLOOPS:
            src, dst = make_graph_no_loops(4,0)
        else:
            src = src_list(4)
            dst = dst_list(4)
        gtemp = dgl.graph((src,dst))
        #print(g.num_nodes())
        gs.append(gtemp)
    #print(len(gs))
    #print(g.num_nodes())
    roll_g = dgl.batch(gs)
    
    
    metrics={"train_sum":0,  "train_roll":0, "train_vec" :0, "train_h" :0,
             "test_sum":0,  "test_roll":0, "test_vec" :0, "test_h" :0}
        

    container = torch.zeros(8,EPOCHS) #["train loss, train grad loss, test loss, test grad loss"]

    
    #wandb.watch(model,log='all')
    
    for epoch in tqdm(range(EPOCHS)):
        model.train()
        
        print("TRAIN")
        
        for train_sample in tqdm(trainset):
            #print(train_sample.num_nodes())
        
            loss=0
            lossH=0
            lossvec=0
            lossroll=0
            opti.zero_grad()
            #model.set_graph(train_sample)
            x_tr = train_sample.ndata["x"].transpose(0,1)
            dx_tr = train_sample.ndata["dx"].transpose(0,1)
            #h_tr = train_sample.ndata["h"].transpose(0,1)
            h_tr = correct_ham_data(train_sample)
    
            x0 = x_tr[0,:,:].requires_grad_()
            #print(x_tr.shape)
            #print(dx_tr.shape)
            #print(h_tr.shape)
            #model.change_graph(roll_g)
            #print(roll_g)
            #x_tr_flat = x_tr.reshape(-1,2)
            model.change_graph(train_sample)
            x_pred, dx_pred, h_pred = model(ts,x0)
            #print(x_tr_flat.shape)
            #h_pred = model(x_tr_flat)
            #print(h_pred.shape)
            #print(h_tr.reshape(-1,1).shape)
            lossH = lossfn(h_pred.flatten(),h_tr.flatten())
            lossvec = lossfn(dx_pred[:,:,0],dx_tr[:,:,0])+lossfn(dx_pred[:,:,1],dx_tr[:,:,1])
            lossroll = lossfn(x_pred[:,:,0],x_tr[:,:,0])+lossfn(x_pred[:,:,1],x_tr[:,:,1])

            #print("after h roll")
            loss += s_alpha[0]* lossroll
            loss += s_alpha[1]* lossH
            loss += s_alpha[2]*lossvec
            
            container[0,epoch] += loss.item()
            container[1,epoch]+=lossroll.item()
            container[2,epoch]+=lossvec.item()
            container[3,epoch]+=lossH.item()
            
            loss.backward()
            opti.step()
        container[0:4,epoch]/=N_train
        print("TEST")
        model.eval()
        
        
        for test_sample in tqdm(testset):
            losst=0
            lossHt=0
            lossvect=0
            lossROLLt=0
            model.change_graph(test_sample)
            x_ts = test_sample.ndata["x"].transpose(0,1)
            dx_ts = test_sample.ndata["dx"].transpose(0,1)
            h_ts =  correct_ham_data(test_sample)
            #print(h_ts)
            #print(h_ts.shape)
            model.change_graph(test_sample)
            x_ts = x_ts.requires_grad_()
            x0 = x_ts[0,:,:]
            #x_ts_flat = x_ts.reshape(-1,2)
            #h_pred = model(x_ts_flat)
            x_pred,dx_pred,h_pred =model(ts,x0.requires_grad_()) 
            lossHt = lossfn(h_pred.flatten(),h_ts.flatten())

            lossROLLt = lossfn(x_pred[:,:,0],x_ts[:,:,0])+lossfn(x_pred[:,:,1],x_ts[:,:,1])
            lossvect = lossfn(dx_pred[:,:,0],dx_ts[:,:,0])+lossfn(dx_pred[:,:,1],dx_ts[:,:,1])
        
            
            losst+=s_alpha[0] * lossROLLt
            losst+=s_alpha[1] * lossHt
            losst+=s_alpha[2] * lossvect 
                
           
            container[4,epoch]+=losst.item()
            container[5,epoch]+=lossROLLt.item()
            container[6,epoch]+=lossvect.item()
            container[7,epoch] += lossHt.item()
        container[4:8,epoch]/=N_test
    
        metrics["train_sum"] = container[0,epoch]
        metrics["train_roll"] = container[1,epoch]
        metrics["train_vec"] = container[2,epoch]
        metrics["train_h"] = container[3,epoch]
        metrics["test_sum"] = container[4,epoch]
        metrics["test_roll"] = container[5,epoch]
        metrics["test_vec"] = container[6,epoch]
        metrics["test_h"] = container[7,epoch]
        #wandb.log(metrics)
            #wandb.log_artifact(model)
        
        print("Epoch: {}\nLOSS: train: {:.6f} roll: {:.6f} vec: {:.6f}  ham: {:.6f} \n".format(epoch+1,
                                                                                            container[0,epoch],
                                                                                            container[1,epoch],
                                                                                            container[2,epoch],
                                                                                            container[3,epoch])+
                                "test: {:.6f} roll: {:.6f} vec: {:.6f}  ham: {:.6f}".format(container[4,epoch],
                                                                                            container[5,epoch],
                                                                                            container[6,epoch],
                                                                                            container[7,epoch]))
   
   
    
    visualize_loss("loss of 4dof pendelum",container)
    dic4 = "res_4dof/"
    torch.save(model,dic_base+"/"+dic4+"model.pt")
    torch.save(container,dic_base+"/"+dic4+"losses.pt")
    torch.save(eval,dic_base+"/"+dic4+"eval.pt")
    torch.save(H,dic_base+"/"+dic4+"eval_H.pt")


if __name__ == "__main__":
    
    with open("configs/pend.yaml", 'r') as f:
        configs = yaml.load(f, yaml.Loader)
    new_full(configs)
    
