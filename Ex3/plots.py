# In[]:
import numpy as np
import matplotlib.pyplot as plt

# load all files from images/numpy
# for each file, plot the image and save it to images/plots
# the file name should be the same

# Scenarios
# 1. MLP vs CNN plotted over epochs
# In[]:
mlp_data = np.load("images/numpy/epochs_acc_SVHN_MLP_0.05_SGD.npy")
cnn_data = np.load("images/numpy/epochs_acc_SVHN_CNN_0.05_SGD.npy")

# In[]:
# x_data[:, 0] is time. convert to start at 0
mlp_data[:, 0] = mlp_data[:, 0] - mlp_data[0, 0]
cnn_data[:, 0] = cnn_data[:, 0] - cnn_data[0, 0]

# In[]:
plt.plot(np.arange(1, 31), mlp_data[:, 1], label="MLP (0.05)")
plt.plot(np.arange(1, 31), cnn_data[:, 1], label="CNN (0.05)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig("images/plots/epochs_acc_SVHN_MLP_CNN_0.05_SGD.png")
# delete the plots
plt.clf()

# 2. MLP vs CNN plotted over time
# In[]:
plt.plot(mlp_data[:, 0], mlp_data[:, 1], label="MLP (0.05)")
plt.plot(cnn_data[:, 0], cnn_data[:, 1], label="CNN (0.05)")
plt.xlabel("Time (s)")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig("images/plots/time_acc_SVHN_MLP_CNN_0.05_SGD.png")

# delete the plots
plt.clf()

# In[]:
# 3. All optimizers plotted over epochs for CNN (one plot per optimizer)

sgd_data_1 = np.load("images/numpy/epochs_acc_SVHN_CNN_0.1_SGD.npy")
sgd_data_2 = np.load("images/numpy/epochs_acc_SVHN_CNN_0.05_SGD.npy")
sgd_data_3 = np.load("images/numpy/epochs_acc_SVHN_CNN_0.01_SGD.npy")
sgd_data_4 = np.load("images/numpy/epochs_acc_SVHN_CNN_0.005_SGD.npy")

adam_data_1 = np.load("images/numpy/epochs_acc_SVHN_CNN_0.1_Adam.npy")
adam_data_2 = np.load("images/numpy/epochs_acc_SVHN_CNN_0.01_Adam.npy")
adam_data_3 = np.load("images/numpy/epochs_acc_SVHN_CNN_0.001_Adam.npy")

rmsprop_data_1 = np.load("images/numpy/epochs_acc_SVHN_CNN_0.1_RMSprop.npy")
rmsprop_data_2 = np.load("images/numpy/epochs_acc_SVHN_CNN_0.01_RMSprop.npy")
rmsprop_data_3 = np.load("images/numpy/epochs_acc_SVHN_CNN_0.001_RMSprop.npy")

plt.plot(np.arange(1, 31), sgd_data_1[:, 1], label="SGD 0.1")
plt.plot(np.arange(1, 31), sgd_data_2[:, 1], label="SGD 0.05")
plt.plot(np.arange(1, 31), sgd_data_3[:, 1], label="SGD 0.01")
plt.plot(np.arange(1, 31), sgd_data_4[:, 1], label="SGD 0.005")

plt.plot(np.arange(1, 31), adam_data_1[:, 1], label="Adam 0.1")
plt.plot(np.arange(1, 31), adam_data_2[:, 1], label="Adam 0.01")
plt.plot(np.arange(1, 31), adam_data_3[:, 1], label="Adam 0.001")

plt.plot(np.arange(1, 31), rmsprop_data_1[:, 1], label="RMSprop 0.1")
plt.plot(np.arange(1, 31), rmsprop_data_2[:, 1], label="RMSprop 0.01")
plt.plot(np.arange(1, 31), rmsprop_data_3[:, 1], label="RMSprop 0.001")

plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig("images/plots/epochs_acc_SVHN_CNN_all_optimizers.png")
# delete the plots
plt.clf()

# In[]:
# 4. Best learning rate for each optimizer plotted over epochs

plt.plot(np.arange(1, 31), sgd_data_2[:, 1], label="SGD 0.05")
plt.plot(np.arange(1, 31), adam_data_3[:, 1], label="Adam 0.001")
plt.plot(np.arange(1, 31), rmsprop_data_3[:, 1], label="RMSprop 0.001")

plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig("images/plots/epochs_acc_SVHN_CNN_best_lr.png")

# %%

# In[]:
# MNIST MLP CPU vs GPU over time
import numpy as np
import matplotlib.pyplot as plt

mnist_mlp_cpu = np.load("images/numpy/epochs_acc_MNIST_MLP_0.1_SGD_cpu.npy")
mnist_mlp_gpu = np.load("images/numpy/epochs_acc_MNIST_MLP_0.1_SGD_gpu.npy")

# x_data[:, 0] is time. convert to start at 0
mnist_mlp_cpu[:, 0] = mnist_mlp_cpu[:, 0] - mnist_mlp_cpu[0, 0]
mnist_mlp_gpu[:, 0] = mnist_mlp_gpu[:, 0] - mnist_mlp_gpu[0, 0]

plt.plot(mnist_mlp_cpu[:, 0], mnist_mlp_cpu[:, 1], label="CPU")
plt.plot(mnist_mlp_gpu[:, 0], mnist_mlp_gpu[:, 1], label="GPU")
plt.xlabel("Time (s)")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig("images/plots/time_acc_MNIST_MLP_CPU_GPU.png")

# %%
