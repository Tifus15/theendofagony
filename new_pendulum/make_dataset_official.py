import torch
from dof3_pendelum_torch import *
from dof2_pendelum_torch import *
from dof1_pendelum_torch import *
from dof4_pendelum_torch import *
from device_util import ROOT_PATH, DEVICE
def angle_transformer(data):
    print(data.shape)
    qp=torch.split(data,int(data.shape[-1]/2),dim=-1)
    x = torch.cos(qp[0])
    y = torch.sin(qp[0])
    out = torch.atan2(y,x)
    print("p shape:{}".format(qp[1].shape))
    print(out.shape)
    return torch.cat((out,qp[1]),dim=-1)


def pendelum_dataset(samples,
                     T,
                     steps,
                     data = [1,1,1,1,1,1,1,1,9.81],
                     range_of_angles=[-torch.pi,torch.pi]):
    dim = 4
    data_maker1 = pendelum1dof(data[0],data[1],data[8])
    data_maker2 = pendelum2dof(data[0],data[1],data[2],data[3],data[8])
    data_maker3 = pendelum3dof(data[0],data[1],data[2],data[3],data[4],data[5],data[8])
    data_maker4 = pendelum4dof(data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8])
    traj_catcher1 = []
    traj_catcher2 = []
    traj_catcher3 = []
    traj_catcher4 = []
    h_catcher1 = []
    h_catcher2 = []
    h_catcher3 = []
    h_catcher4 = []

    inits = torch.cat((range_of_angles[0]+ torch.rand(samples,1,dim)*(range_of_angles[1]-range_of_angles[0]),
                    + torch.zeros(samples,1,dim)),dim=-1)

    print(inits.shape)
    t = torch.linspace(0,T,steps)
    count=0
    i=0
    repeat = False
    while i < samples:
        print("#################\nI: {}".format(i))
        count 
        count = 0
        flag = False
        if repeat:
            inits[i,:,:] = torch.cat((range_of_angles[0]+ torch.rand(1,1,dim)*(range_of_angles[1]-range_of_angles[0]),
                    + torch.zeros(1,1,dim)),dim=-1)
            repeat = False 

        print("Dof1")
        init1 = inits[i,:,[0,4]]
        H = dof1_hamiltonian_eval(data_maker1,init1.unsqueeze(0))
       
        trajectories1 = dof1_dataset(data_maker1.to(DEVICE),t.to(DEVICE),init1.unsqueeze(0).to(DEVICE))
        max_q1 = torch.max(trajectories1[:,0,0,0])
        min_q1 = torch.max(trajectories1[:,0,0,0])

        if max_q1 > torch.pi or min_q1 < -torch.pi or torch.abs(H[0]) < 1:
            repeat = True
            print("repeat")
            continue
        else:
            count+=1


        print("Dof2")
        init2 = inits[i,:,[0,1,4,5]]
        H = dof2_hamiltonian_eval(data_maker2,init2.unsqueeze(0))
        
       
    
        trajectories2 = dof2_dataset(data_maker2.to(DEVICE),t.to(DEVICE),init2.unsqueeze(0).to(DEVICE))
        max_q1 = torch.max(trajectories2[:,0,0,0])
        min_q1 = torch.max(trajectories2[:,0,0,0])
        max_q2 = torch.max(trajectories2[:,0,0,1])
        min_q2 = torch.max(trajectories2[:,0,0,1])

        if max_q1 > torch.pi or min_q1 < -torch.pi or max_q2 > torch.pi or min_q2 < -torch.pi or torch.abs(H[0]) < 1:
            repeat = True
            print("repeat")
            continue
        else:
            count+=1
        
        print("Do3")
        init3 = inits[i,:,[0,1,2,4,5,6]]
        H = dof3_hamiltonian_eval(data_maker3,init3.unsqueeze(0))
       
        
        
        trajectories3 = dof3_dataset(data_maker3.to(DEVICE),t.to(DEVICE),init3.unsqueeze(0).to(DEVICE))
        max_q1 = torch.max(trajectories3[:,0,0,0])
        min_q1 = torch.max(trajectories3[:,0,0,0])
        max_q2 = torch.max(trajectories3[:,0,0,1])
        min_q2 = torch.max(trajectories3[:,0,0,1])
        max_q3 = torch.max(trajectories3[:,0,0,2])
        min_q3 = torch.max(trajectories3[:,0,0,2])

        if max_q1 > torch.pi or min_q1 < -torch.pi or max_q2 > torch.pi or min_q2 < -torch.pi or max_q3 > torch.pi or min_q3 < -torch.pi or torch.abs(H[0]) < 1:
            repeat = True
            print("repeat")
            continue
        else:
            count+=1
        print("Dof4")
        init4 = inits[i,:,:]
        H = dof4_hamiltonian_eval(data_maker4,init4.unsqueeze(0))
        
        
    
        trajectories4 = dof4_dataset(data_maker4.to(DEVICE),t.to(DEVICE),init4.unsqueeze(0).to(DEVICE))
        max_q1 = torch.max(trajectories4[:,0,0,0])
        min_q1 = torch.max(trajectories4[:,0,0,0])
        max_q2 = torch.max(trajectories4[:,0,0,1])
        min_q2 = torch.max(trajectories4[:,0,0,1])
        max_q3 = torch.max(trajectories4[:,0,0,2])
        min_q3 = torch.max(trajectories4[:,0,0,2])
        max_q4 = torch.max(trajectories4[:,0,0,3])
        min_q4 = torch.max(trajectories4[:,0,0,3])

        if max_q1 > torch.pi or min_q1 < -torch.pi or max_q2 > torch.pi or min_q2 < -torch.pi or max_q3 > torch.pi or min_q3 < -torch.pi or max_q4 > torch.pi or min_q4 < -torch.pi or torch.abs(H[0]) < 1:
            repeat = True
            print("repeat")
            continue
        else:
            count+=1

        if count == 4:
            traj_catcher1.append(trajectories1)
            traj_catcher2.append(trajectories2)
            traj_catcher3.append(trajectories3)
            traj_catcher4.append(trajectories4)
            H1 = dof1_hamiltonian_eval(data_maker1,trajectories1[:,0,:,0:2])
            H2 = dof1_hamiltonian_eval(data_maker2,trajectories2[:,0,:,0:4])
            H3 = dof1_hamiltonian_eval(data_maker3,trajectories3[:,0,:,0:6])
            H4 = dof1_hamiltonian_eval(data_maker4,trajectories4[:,0,:,0:8])
            print("traj shape {}".format(trajectories4.shape))
            print("H.shape {}".format(H4.shape))
            h_catcher1.append(H1.unsqueeze(1).unsqueeze(1))
            h_catcher2.append(H2.unsqueeze(1).unsqueeze(1))
            h_catcher3.append(H3.unsqueeze(1).unsqueeze(1))
            h_catcher4.append(H4.unsqueeze(1).unsqueeze(1))
            i+=1
    trajvec1 = torch.cat((traj_catcher1),dim=1)
    trajvec2 = torch.cat((traj_catcher2),dim=1)
    trajvec3 = torch.cat((traj_catcher3),dim=1)
    trajvec4 = torch.cat((traj_catcher4),dim=1)
    h1 = torch.cat((h_catcher1),dim=1)
    h2 = torch.cat((h_catcher2),dim=1)
    h3 = torch.cat((h_catcher3),dim=1)
    h4 = torch.cat((h_catcher4),dim=1)
    out1 = torch.cat((trajvec1,h1),dim=-1)
    out2 = torch.cat((trajvec2,h2),dim=-1)
    out3 = torch.cat((trajvec3,h3),dim=-1)
    out4 = torch.cat((trajvec4,h4),dim=-1)

    return out1, out2, out3, out4


out1, out2, out3, out4 =pendelum_dataset(25,
                1.27,
                128,
                data = [1,1,1,1,1,1,1,1,9.81],
                range_of_angles=[-torch.pi,torch.pi])

print(out1.shape)
print(out2.shape)
print(out3.shape)
print(out4.shape)
 
print(out1[:,0,0,-1])
print(out2[:,0,0,-1])
print(out3[:,0,0,-1])
print(out4[:,0,0,-1])

torch.save(out1, "pend_1.pt")
torch.save(out2, "pend_2.pt")
torch.save(out3, "pend_3.pt")
torch.save(out4, "pend_4.pt")
