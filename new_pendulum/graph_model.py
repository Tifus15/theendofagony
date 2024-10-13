import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ghnn_model import *



class combined_layer(nn.Module):
    def __init__(self,g, in_dim, out_dim,bias=True,att_dropout =1.0):
        super(combined_layer, self).__init__()
        self.GCN = GNNlayer(g, in_dim, out_dim,bias = bias)
        self.GAT = GATLayer(g, in_dim, out_dim,bias = bias,dropout = att_dropout)
        self.decoder = nn.Linear(out_dim*2, out_dim)
        
    def forward(self,x):
        y1 = self.GCN(x)
        y2 = self.GAT(x)
        y_raw = torch.cat((y1,y2),dim=-1)
        y = self.decoder(y_raw)
        return y + y1 + y2
    
    def change_graph(self,g):
        self.GCN.change_graph(g)
        self.GAT.change_graph(g)
        
        
class GNN_HNN(nn.Module):
    def __init__(self,g, in_dim,hid_dim, out_dim,acts,bias=True,dropout =0.8):
        super().__init__()
        self.g = g
        self.g_hnn=g
        modules = []
    
        modules.append(combined_layer(g,in_dim,hid_dim,bias=bias,att_dropout = dropout))
        modules.append(function_act(acts[0]))
        for i in range(1,len(acts)-1):
            modules.append(combined_layer(g,hid_dim,hid_dim,bias=bias,att_dropout = dropout))
            if acts[i] != "":
                modules.append(function_act(acts[i]))
        modules.append(combined_layer(g,hid_dim,out_dim,bias=bias,att_dropout = dropout))
        
            
        self.net = nn.Sequential(*modules)
    
    def change_graph(self,g):
        #print("chaning in ghnn")
        for module in self.net.modules():
            if isinstance(module,combined_layer):
                module.change_graph(g)   
        self.g_hnn = g
    
    def reset_graph(self):
        for module in self.net.modules():
            if isinstance(module,combined_layer):
                module.change_graph(self.g)
        self.g_hnn = self.g
    
    """   
    def H(self,y):
        return torch.sum(self.net(y.float()),dim=0)
    """
    def forward(self,x,list=False):
        H_feat = self.net(x.float())
        #print(H_feat.shape)
        self.g_hnn.ndata["temp"] = H_feat
        gs = dgl.unbatch(self.g_hnn)
        #print("from model")
        #print("unbatched {}".format(len(gs)))
        h_list = []
        for g in gs:
            h = g.ndata["temp"]
            #print("h {}".format(h.shape))
            h_sc = h.sum()
           # print(h_sc)
            h_list.append(h_sc.unsqueeze(0))
        out=torch.cat((h_list),dim=0).unsqueeze(0)
        #print(out)
        return out
    
"""
g = dgl.graph(([0,0,1,1],[0,1,0,1]))
model = GNN_HNN(g,2,128,6,["tanh"," "])
print(model)

x = torch.rand(2,2)
print(x)
y = model(x)
print(y)

g = dgl.graph(([0,0,1,1,2,2],[0,1,0,1,0,2]))
model.change_graph(g)
x = torch.rand(3,2).requires_grad_()
y = model(x)
print(y)
dy=torch.autograd.grad(y,x)
print(dy)
"""

class GNN_maker_HNN(nn.Module):
    def __init__(self,g, in_dim,hid_dim, out_dim,acts,bias=True,type="GCN",dropout =1.0):
        super().__init__()
        self.g = g
        self.g_hnn=g
        self.entry = Sin()
        modules = []
        if type == "GCN":
            modules.append(GNNlayer(g,in_dim,hid_dim,bias=bias))
            modules.append(function_act(acts[0]))
            for i in range(1,len(acts)-1):
                modules.append(GNNlayer(g,hid_dim,hid_dim,bias=bias))
                if acts[i] != "":
                    modules.append(function_act(acts[i]))
            modules.append(GNNlayer(g,hid_dim,out_dim,bias=bias))
        elif type == "GAT":
            modules.append(GATLayer(g,in_dim,hid_dim,bias=bias,dropout=dropout))
            modules.append(function_act(acts[0]))
            for i in range(1,len(acts)-1):
                modules.append(GATLayer(g,hid_dim,hid_dim,bias=bias))
                if acts[i] != "":
                    modules.append(function_act(acts[i]))
            modules.append(GATLayer(g,hid_dim,out_dim,bias=bias,dropout=dropout))
        elif type == "PULL":
            modules.append(PulloutLayer(g,in_dim,hid_dim,bias=bias))
            modules.append(function_act(acts[0]))
            for i in range(1,len(acts)-1):
                modules.append(PulloutLayer(g,hid_dim,hid_dim,bias=bias))
                if acts[i] != "":
                    modules.append(function_act(acts[i]))
            modules.append(PulloutLayer(g,hid_dim,out_dim,bias=bias))
            
        self.net = nn.Sequential(*modules)
    
    def change_graph(self,g):
        #print("chaning in ghnn")
        for module in self.net.modules():
            if isinstance(module,GATLayer) or isinstance(module,GNNlayer) or isinstance(module,PulloutLayer):
                module.change_graph(g)   
        self.g_hnn = g
    
    def reset_graph(self):
        for module in self.net.modules():
            if isinstance(module,GATLayer) or isinstance(module,GNNlayer) or isinstance(module,PulloutLayer):
                module.change_graph(self.g)
        self.g_hnn = self.g
    
    """   
    def H(self,y):
        return torch.sum(self.net(y.float()),dim=0)
    """
    def forward(self,x,list=False):
        H_feat = self.net(self.entry(x.float()))
        #print(H_feat.shape)
        self.g_hnn.ndata["temp"] = H_feat
        gs = dgl.unbatch(self.g_hnn)
        #print("from model")
        #print("unbatched {}".format(len(gs)))
        h_list = []
        for g in gs:
            h = g.ndata["temp"]
            #print("h {}".format(h.shape))
            h_sc = h.sum()
           # print(h_sc)
            h_list.append(h_sc.unsqueeze(0))
        out=torch.cat((h_list),dim=0).unsqueeze(0)
        #print(out)
        return out

class rollout_GNN_GRU(nn.Module):
    def __init__(self,g,in_dim,hid,gnn_size,acts,bias=True,type="GCN",dropout=1.0):
        super().__init__()
        self.GRU = nn.GRU(in_dim,hid,1).to(torch.device("cpu"))
        self.GNN = GNN_maker_HNN(g,hid,gnn_size,in_dim,acts,bias=bias,type=type,dropout=dropout).cpu()
        self.g = g
        #self.NN = nn.Linear(hid,in_dim)
        nn.init.normal_(self.GRU.weight_ih_l0,mean=0.,std=0.1)
        nn.init.normal_(self.GRU.weight_hh_l0,mean=0.,std=0.1)
        nn.init.constant_(self.GRU.bias_ih_l0,val=0)
        nn.init.constant_(self.GRU.bias_hh_l0,val=0)
    
    def change_graph(self,g):
        #print("changing in GRU")
        self.GNN.change_graph(g)
       
    
    def reset_graph(self):
        self.GNN.change_graph(self.g)
        
        
    def forward(self,t,x0):
        #print(x0.device)
        #print(t.device)
        #print(self.GNN.g_hnn.device)
        batches =len(dgl.unbatch(self.GNN.g_hnn))
        #print("batches {}".format(batches))
        T = len(t)
        output = torch.zeros(T,x0.shape[0],x0.shape[1])
        doutput = torch.zeros(T,x0.shape[0],x0.shape[1])
        houtput = torch.zeros(T,batches)
        #print(output.device)
        #print(doutput.device)
        #print(houtput.device)
        #print(self.GRU.device)
        #print(self.GNN.device)
        output[0,:,:] = x0
        xi, hidden = self.GRU(x0.unsqueeze(dim=0))
        #print("x0: {}".format(x0.shape))
        #print("xi: {}".format(xi.unsqueeze(dim=0).shape))
        hi = self.GNN(xi.squeeze())
        #print("hi {}".format(hi.shape))
        houtput[0,:]=hi
        H_l = torch.split(houtput[0,:],1,dim=0)
        dHdx = torch.autograd.grad(H_l,x0,retain_graph=True, create_graph=True)[0] 
        dqdp_s = torch.split(dHdx,2,dim=-1)
        dx_pred = torch.cat((dqdp_s[1],-dqdp_s[0]),dim=-1)
        doutput[0,:,:] = dx_pred
        #print("xii: {}".format(xii.shape))
        #print("hidden: {}".format(hidden.shape))
        #print(xii.shape)
        
        dt = t[1]-t[0]
        temp = output[0,:,:] + dt*doutput[0,:,:]
        output[1,:,:] = temp
        for i in range(2,T):
            #print("xi: {}".format(xi.unsqueeze(dim=0).shape))
            xi, hidden = self.GRU(temp.unsqueeze(dim=0),hidden)
            #print("hidden: {}".format(hidden.shape))
            hi = self.GNN(xi.squeeze())
            houtput[i-1,:]=hi
            H_l = torch.split(houtput[i-1,:],1,dim=0)
            dHdx = torch.autograd.grad(H_l,temp,retain_graph=True, create_graph=True)[0] 
            dqdp_s = torch.split(dHdx,2,dim=-1)
            dx_pred = torch.cat((dqdp_s[1],-dqdp_s[0]),dim=-1)
            
            
            
            doutput[i-1,:,:] = dx_pred
            #print("xii: {}".format(xii.shape))
            
            dt = t[i]-t[i-1]
            temp = output[i-1,:,:]+ dt*dx_pred
            output[i,:,:] = temp
        xi, hidden = self.GRU(temp.unsqueeze(dim=0),hidden)
        #print("hidden: {}".format(hidden.shape))
        hi = self.GNN(xi.squeeze())
        houtput[-1,:] = hi
        H_l = torch.split(houtput[-1,:],1,dim=0)
        dHdx = torch.autograd.grad(H_l,temp,retain_graph=True, create_graph=True)[0] 
        dqdp_s = torch.split(dHdx,2,dim=-1)
        dx_pred = torch.cat((dqdp_s[1],-dqdp_s[0]),dim=-1)
        doutput[-1,:,:] = dx_pred
        return  output, doutput, houtput