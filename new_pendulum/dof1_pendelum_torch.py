import torch
import torch.nn as nn
from torch.autograd.functional import hessian
from torch.autograd import grad
from time import time
from torchdiffeq import odeint 
from tqdm import tqdm
# inputs: thera_dtheta - torch tensor(1,2) = [theta, dtheta] - state, vel
#         thera_ptheta - torch tensor(1,2) = [theta, ptheta] - state, mom


class pendelum1dof(nn.Module):
    def __init__(self,m,l,g):
        super(pendelum1dof,self).__init__()
        self.m = m
        self.l = l
        self.g = g
    # position of the joint in cartesian space 
    def cart_traj(self,theta_dtheta):
        x = self.l*torch.sin(theta_dtheta[:,0:1])
        y = -self.l*torch.cos(theta_dtheta[:,0:1])
        return torch.cat((x,y),dim=-1)
    # linear velocity of the joint in cartesian space
    def cart_der(self,theta_dtheta):
        dx = theta_dtheta[:,1:2]*self.l*torch.cos(theta_dtheta[:,0:1])
        dy = theta_dtheta[:,1:2]*self.l*torch.sin(theta_dtheta[:,0:1])
        return torch.cat((dx,dy),dim=-1)
    # squared linear velocity norm |v|**2
    def cart_sq_der(self,theta_dtheta):
        vec = self.cart_der(theta_dtheta)
        return vec[:,0:1].pow(2) + vec[:,1:2].pow(2)
    # kinetic energy(q,dq) 
    def kin(self,theta_dtheta):
        vs = self.cart_sq_der(theta_dtheta)
        return 0.5* self.m * vs
    # potential energy(q)
    def pot(self,theta_ptheta):
        traj = self.cart_traj(theta_ptheta)
        return self.m*self.g*(traj[0,1:2])
    # matrix M for p = dq @ M - matrix is in this case a scalar
    def trans_matrix(self,theta_dtheta):
        Hess = hessian(self.kin,theta_dtheta,create_graph=True)
        Hess=Hess.reshape(2,2) 
        return Hess[1,1]
    # p = dq @ M
    def d_theta_2_p_theta(self,theta_dtheta):
        Hess = self.trans_matrix(theta_dtheta)
        theta_ptheta = theta_dtheta.clone()
        theta_ptheta[:,1:2] = Hess * theta_dtheta[:,1:2] 
        return theta_ptheta
    #  dq =  p @ inv(M) 
    def p_theta_2_d_theta(self,theta_ptheta):
        Hess = self.trans_matrix(theta_ptheta)
        theta_dtheta = theta_ptheta.clone()
        theta_dtheta[:,1:2] = (1/Hess) * theta_ptheta[:,1:2] 
        return theta_dtheta

    # hamiltonian = Kinetik(q,p) + Potential(q)
    def H(self,theta_ptheta):
        K = self.kin(self.p_theta_2_d_theta(theta_ptheta))
        U = self.pot(theta_ptheta)
        return K+U
    # hamiltonian gradient for canonical coordinates [dH_dtheta, dH_dptheta]
    def dH_dth_dpth(self,theta_ptheta):
        Ene = self.H(theta_ptheta)
        dEne= torch.autograd.grad(Ene,theta_ptheta,create_graph=True)
        return dEne[0]
    # equations:
    # dtheta = dH_dptheta
    # dp = - dH_dtheta
    def forward(self,t,y):
        dHdx = self.dH_dth_dpth(y.requires_grad_())
        dq = dHdx[:,1:2]
        dp = -dHdx[:,0:1]
        out = torch.cat((dq,dp),dim=-1).detach()
        out.requires_grad=False
        return out.detach()
    
def dof1_hamiltonian_eval(model,trajectory):
    H = torch.Tensor(trajectory.shape[0],1)
    for i in range(trajectory.shape[0]):
        H[i,:] = model.H(trajectory[i,:,:])
    return H.detach()

def dof1_grads(model,traj):
    list = []
    for i in range(traj.shape[0]):
        x0 = traj[i,:,:].view(1,-1)
       #print(x0.shape)
        dx = model.forward(0,x0)
        list.append(dx.unsqueeze(0))
    return torch.cat((list),dim=0) 
      
    
def dof1_trajectory(model,t,y0,method="rk4"):
    # y0 -[1,6]
    print(y0.shape)
    out = odeint(model,y0,t,method=method)
    return out

def dof1_dataset(model,t,y0,method="rk4"):
    # y0 - [b,1,2]
    list=[]
    for i in tqdm(range(y0.shape[0])):
        x = dof1_trajectory(model,t,y0[i,:,:],method).unsqueeze(1)
        dx = dof1_grads(model,x).unsqueeze(1)
        print(dx.shape)
        traj = torch.cat((x,dx),dim=-1)
        list.append(traj)
    
    return torch.cat((list),dim=1)

def dof1_symplectic_trajectory(model,t,y0,method="euler"):
    out = torch.Tensor(len(t),y0.shape[0],y0.shape[1])
    out[0,:,:] = y0
    for i in range(1,len(t)):
        #print(out[i-1,:,:])
        dt = t[i]-t[i-1]
        temp = out[i-1,:,:]
        if method =="euler":
            dp = model.forward(0,temp)[:,1:2]
            temp.requires_grad = False
            temp[:,1:2] = temp[:,1:2] + dt*dp
            dq = model.forward(0,temp)[:,0:1]
            temp.requires_grad = False
        
            temp[:,0:1] = temp[:,0:1] + dt*dq
        out[i,:,:] = temp
    return out.detach()

"""
model = pendelum1dof(1,1,10)
theta_ptheta = torch.Tensor([[1.5,-1.4]])
print(theta_ptheta)
#print(model.forward(0,theta_ptheta))
t = torch.linspace(0,1,100)
traj=dof1_trajectory(model,t,theta_ptheta,method="rk4")
print(traj)
#traj=dof1_symplectic_trajectory(model,t,theta_ptheta,method="euler")
print(dof1_hamiltonian_eval(model,traj))
#print(odeint(model,theta_ptheta,t,method="rk4"))
"""
