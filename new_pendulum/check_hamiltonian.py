import torch

x1 = torch.load("pend_1.pt")
x2 = torch.load("pend_2.pt")
x3 = torch.load("pend_3.pt")
x4 = torch.load("pend_4.pt")
print(x1.shape)

H1 = x1[:,:,:,-1].squeeze()
H2 = x2[:,:,:,-1].squeeze()
H3 = x3[:,:,:,-1].squeeze()
H4 = x4[:,:,:,-1].squeeze()

H1_mean = torch.mean(H1,dim=0)
H2_mean = torch.mean(H2,dim=0)
H3_mean = torch.mean(H3,dim=0)
H4_mean = torch.mean(H4,dim=0)

H1_std = torch.std(H1,dim=0)
H2_std = torch.std(H2,dim=0)
H3_std = torch.std(H3,dim=0)
H4_std = torch.std(H4,dim=0)

eval1 =torch.abs(H1_std/H1_mean)
eval2 =torch.abs(H2_std/H2_mean)
eval3 =torch.abs(H3_std/H3_mean)
eval4 =torch.abs(H4_std/H4_mean)

print(torch.max(eval1))
print(torch.max(eval2))
print(torch.max(eval3))
print(torch.max(eval4))




