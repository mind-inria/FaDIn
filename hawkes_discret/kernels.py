import torch
from hawkes_discret.utils.utils import optimizer

class KernelExpDiscret(object):
    """"""
    def __init__(self, decay, upper,  discret_step):
        """ baseline: vecteur de taille dim (mu)
            decay: matrice de taille dim x dim (gamma)           
            end_time: int, borne supérieure du processus 
            upper: float, right bound of the support of the kernel
            size_discret: int, taille de la grille de discrétisation du kernel
            size_grid: int, taille total de la grille de discrétisation"""

        self.decay = decay
        self.dim = decay.shape[0]
        self.size_discret = int(1 / discret_step)
        self.upper = upper


    def eval(self, time):
        """Return kernel evaluate on the discretisation grid: time
        kernel_values:  tensor de taille (dim x dim x len(time))"""

        self.kernel_values = self.decay.unsqueeze(2) * torch.exp(-self.decay.unsqueeze(2)*time)
        self.mask_kernel = (time <= 0) | (time > self.upper)
        self.kernel_values[:, :, self.mask_kernel] = 0.
        self.kernel_values /= self.kernel_values.sum(2)[:, :, None] # intéret de * dt ?

        return self.kernel_values

    def grad_params(self, time, autodiff=False):
        """Return grad w.r.t. decay of size (dim x dim x len(time))
            """
        if autodiff:
            return 0 #en construction
        else:
            grad = (1-self.decay.unsqueeze(2)*time) * torch.exp(-self.decay.unsqueeze(2)*time)
            grad[:, :, self.mask_kernel] = 0.
            return grad

    def integrate(self):
        """indices_grid: dim x taille de la grille
            return matrix of approximated integral of size (dim x dim) (equal to one)"""
        return self.kernel_values.sum(2)


    def intensity_eval(self, baseline, adjacency, events_bool):
        """ return the evaluation of the intensity in each point of the grid
            vector of size: dim x size_grid
        """
        self.size_grid = events_bool[0].shape[0]
        kernel_values_adj = self.kernel_values * adjacency[:, :, None]

        #ancienne version
        #intensity_ = torch.zeros(self.dim, self.size_grid)  
        #for i in range(self.dim):

        #    intensity_[i] = baseline[i] + torch.conv_transpose1d(
        #        events_bool[i].view(1, self.size_grid),
        #        kernel_values_adj[i].view(1, self.dim, self.size_discret))[:, :-self.size_discret+1].sum(0)

        intensity = torch.zeros(self.dim, self.dim, self.size_grid)
        for i in range(self.dim):
            intensity[i, :, :] = torch.conv_transpose1d(
                    events_bool[i].view(1, self.size_grid),
                    kernel_values_adj[:, i].view(1, self.dim, self.size_discret))[:, :-self.size_discret+1]

        return intensity.sum(0) + baseline.unsqueeze(1)

#integral en continue
#    def integrate(self, start_time, end_time):
#        return torch.exp(- self.decay * start_time) - torch.exp(- self.decay * end_time)