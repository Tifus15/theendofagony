import torch
import matplotlib.pyplot as plt

epochs=300
epochs_t = torch.linspace(1,epochs,epochs)

loss3 = torch.load("res_3dof/losses.pt").transpose(0,1)
loss4 = torch.load("res_4dof/losses.pt").transpose(0,1)
print(loss3.shape)

fig, ax  = plt.subplots(1,4)
ax[0].set_title("Train/Test Loss")
ax[0].set_xlabel("epochs")
ax[0].set_ylabel("loss")

ax[1].set_title("roll")
ax[1].set_xlabel("epochs")
ax[1].set_ylabel("loss")

ax[2].set_title("vec")
ax[2].set_xlabel("epochs")
ax[2].set_ylabel("loss")

ax[3].set_title("h")
ax[3].set_xlabel("epochs")
ax[3].set_ylabel("loss")


ax[0].semilogy(epochs_t,loss3[:,0],c="b")
ax[0].semilogy(epochs_t,loss3[:,4],c="r")

ax[1].semilogy(epochs_t,loss3[:,1],c="b")
ax[1].semilogy(epochs_t,loss3[:,5],c="r")

ax[2].semilogy(epochs_t,loss3[:,2],c="b")
ax[2].semilogy(epochs_t,loss3[:,6],c="r")

ax[3].semilogy(epochs_t,loss3[:,3],c="b")
ax[3].semilogy(epochs_t,loss3[:,7],c="r")


ax[0].legend(["train loss","test_loss"])
ax[1].legend(["train loss","test_loss"])
ax[2].legend(["train loss","test_loss"])
ax[3].legend(["train loss","test_loss"])
fig.suptitle("3dof case")

fig, ax  = plt.subplots(1,4)
ax[0].set_title("Train/Test Loss")
ax[0].set_xlabel("epochs")
ax[0].set_ylabel("loss")

ax[1].set_title("roll")
ax[1].set_xlabel("epochs")
ax[1].set_ylabel("loss")

ax[2].set_title("vec")
ax[2].set_xlabel("epochs")
ax[2].set_ylabel("loss")

ax[3].set_title("h")
ax[3].set_xlabel("epochs")
ax[3].set_ylabel("loss")


ax[0].semilogy(epochs_t,loss4[:,0],c="b")
ax[0].semilogy(epochs_t,loss4[:,4],c="r")

ax[1].semilogy(epochs_t,loss4[:,1],c="b")
ax[1].semilogy(epochs_t,loss4[:,5],c="r")

ax[2].semilogy(epochs_t,loss4[:,2],c="b")
ax[2].semilogy(epochs_t,loss4[:,6],c="r")

ax[3].semilogy(epochs_t,loss4[:,3],c="b")
ax[3].semilogy(epochs_t,loss4[:,7],c="r")


ax[0].legend(["train loss","test_loss"])
ax[1].legend(["train loss","test_loss"])
ax[2].legend(["train loss","test_loss"])
ax[3].legend(["train loss","test_loss"])
fig.suptitle("4dof case")
plt.show()