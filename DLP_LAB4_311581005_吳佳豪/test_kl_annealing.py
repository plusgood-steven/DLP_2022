#%%
import matplotlib.pyplot as plt
import numpy as np

class kl_annealing():
    def __init__(self, ratio,cycle,niter):
        super().__init__()
        self.ratio = ratio
        self.cycle = cycle
        self.period = niter / self.cycle
        self.step = 1 / (self.period * self.ratio)
        self.v = 0
        self.i = 0

    def update(self):
        self.v += self.step
        self.i += 1
        if self.v >= 1:
            if self.i >= self.period:
                self.v = 0
                self.i = 0
            else :
                self.v = 1
    
    def get_beta(self):
        return self.v

n_epoch = 300
annealing = kl_annealing(1,3,n_epoch)
beta_np_cyc = np.zeros(n_epoch)
for i in range(n_epoch):
    beta_np_cyc[i] = annealing.get_beta()
    annealing.update()

fig=plt.figure(figsize=(8,4.0))
stride = max( int(n_epoch / 8), 1)

plt.plot(range(n_epoch), beta_np_cyc, '-', label='Cyclical', marker= 's', color='k', markevery=stride,lw=2,  mec='k', mew=1 , markersize=10)
plt.show()
#%%