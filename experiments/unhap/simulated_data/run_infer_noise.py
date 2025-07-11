# %% import stuff
import numpy as np
import itertools
from joblib import Parallel, delayed
import pandas as pd
import torch
import time

from fadin.utils.utils_simu import simu_marked_hawkes_cluster
from fadin.utils.utils_simu import simu_multi_poisson
from fadin.solver import UNHaP
from fadin.kernels import DiscreteKernelFiniteSupport
from fadin.utils.utils import projected_grid_marked, optimizer_fadin
from fadin.init import momentmatching_fadin, momentmatching_unhap
from fadin.utils.compute_constants import get_zG, get_ztzG_approx, get_ztzG
from fadin.loss_and_gradient import discrete_l2_loss_precomputation
from fadin.loss_and_gradient import discrete_ll_loss_conv
from fadin.loss_and_gradient import discrete_l2_loss_conv


# %% Define JointFaDIn and JointFaDInDenoising solvers

def get_zN_joint(events_grid, events_ground_grid, L):
    """
    events_grid.shape = n_dim, n_grid
    zN.shape = n_dim, n_dim, L
    zLN.shape = n_dim, n_dim
    """
    n_dim, _ = events_grid.shape

    zN = np.zeros(shape=(n_dim, n_dim, L))
    for i in range(n_dim):
        ei = events_grid[i]  # Count drived marks
        for j in range(n_dim):
            ej = events_grid[j]
            zN[i, j, 0] = ej @ ei
            for tau in range(1, L):
                zN[i, j, tau] = ei[tau:] @ ej[:-tau]

    return 2 * zN  # to get a density for the distribution of the mark


def get_grad_baseline_joint(zG, baseline, alpha, kernel,
                            delta, end_time, sum_marks,
                            square_int_marks, n_ground_events):
    """Return the gradient of the discrete l2 loss w.r.t. the baseline.

    .. math::
        N_T\\frac{\\partial\\mathcal{L}_G}{\\partial \\mu_{m}} =
        2 T \\mu_m -  2 N_T^m + 2 \\Delta\\sum_{j=1}^{p} \\sum_{\\tau=1}^{L}
        \\phi_{mj}^\\Delta[\\tau]\\Phi_{j}(\\tau; G)

    Parameters
    ----------
    zG : tensor, shape (n_dim, L)

    baseline : tensor, shape (n_dim,)
        Baseline parameter of the intensity of the Hawkes process.

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L)
        Kernel values on the discretization.

    delta : float
        Step size of the discretization grid.

    sum_marks : tensor, shape (n_dim,)
        Sum of marks for each dimension.

    end_time : float
        The end time of the Hawkes process.

    n_ground_events : tensor, shape (n_dim,)
        Number of events for each dimension.

    Returns
    ----------
    grad_baseline: tensor, shape (dim,)
    """
    n_dim, _, _ = kernel.shape

    if n_dim > 1:
        cst1 = end_time * baseline * square_int_marks
        cst2 = 0.5 * n_ground_events.sum()

        dot_kernel = torch.einsum('kju,ju->kj', kernel, zG)
        dot_kernel_ = (dot_kernel * alpha).sum(1) * square_int_marks

        grad_baseline = (dot_kernel_ * delta + cst1 - sum_marks) / cst2
    else:
        grad_baseline_ = torch.zeros(n_dim)
        for k in range(n_dim):
            temp = 0
            for j in range(n_dim):
                temp += alpha[k, j] * (zG[j] @ kernel[k, j])
            grad_baseline_[k] = delta * temp * square_int_marks[k]
            grad_baseline_[k] += end_time * baseline[k] * square_int_marks[k]
            grad_baseline_[k] -= 2 * sum_marks[k]  # add of joint MarkedFaDIn
        grad_baseline = 2 * grad_baseline_ / n_ground_events.sum()

    return grad_baseline


def get_grad_alpha_joint(zG, zN, ztzG, baseline, alpha, kernel, delta,
                         square_int_marks, n_ground_events):
    """Return the gradient of the discrete l2 loss w.r.t. alpha.

    .. math::
        N_T\\frac{\\partial\\mathcal{L}_G}{\\partial \\alpha_{ml}} =
        2\\Delta \\mu_m  \\sum_{\\tau=1}^{L} \\frac{\\partial
        \\phi_{m,l}^\\Delta[\\tau]}{\\partial \\alpha_{m,l}} \\Phi_l(\\tau; G)
        + 2 \\Delta \\sum_{k=1}^{p} \\sum_{\\tau=1}^{L} \\sum_{\\tau'=1}^{L}
        \\phi_{mk}^\\Delta[\\tau'] \\frac{\\partial \\phi_{m,l}^\\Delta[\\tau]}
        {\\partial \\alpha_{m,l}} \\Psi_{l,k}(\\tau, \\tau'; G)

    Parameters
    ----------
    zG : tensor, shape (n_dim, L)

    zN : tensor, shape (n_dim, L)

    ztzG : tensor, shape (n_dim, n_dim, L, L)

    baseline : tensor, shape (n_dim,)
        Baseline parameter of the intensity of the Hawkes process.

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L)
        Kernel values on the discretization.

    delta : float
        Step size of the discretization grid.

    n_ground_events : tensor, shape (n_dim,)
        Number of events for each dimension.

    Returns
    ----------
    grad_alpha : tensor, shape (n_dim, n_dim)
    """
    n_dim, _, _ = kernel.shape

    if n_dim > 1:
        cst1 = delta * baseline.view(n_dim, 1)

        dot_kernel = torch.einsum('njuv,knu->njkv', ztzG, kernel)
        ker_ztzg = torch.einsum('kju,njku->knj', kernel, dot_kernel)

        term1 = torch.einsum('knj,kj->kn', ker_ztzg, alpha)
        term2 = torch.einsum('knu,nu->kn', kernel, zG)
        term3 = torch.einsum('knu,knu->kn', zN, kernel)

        grad_alpha_ = square_int_marks.unsqueeze(1) * term1 * delta + \
            square_int_marks.unsqueeze(1) * cst1 * term2 - term3
    else:
        grad_alpha_ = torch.zeros(n_dim, n_dim)
        for k in range(n_dim):
            dk = delta * baseline[k]
            for n in range(n_dim):
                temp = 0
                for j in range(n_dim):
                    temp += alpha[k, j] * (torch.outer(kernel[k, n], kernel[k, j]) *
                                           ztzG[n, j]).sum()
                grad_alpha_[k, n] += delta * temp * square_int_marks[k]
                grad_alpha_[k, n] += dk * kernel[k, n] @ zG[n] * square_int_marks[k]
                grad_alpha_[k, n] -= zN[k, n] @ kernel[k, n]

    grad_alpha = 2 * grad_alpha_ / n_ground_events.sum()

    return grad_alpha


def get_grad_eta_joint(zG, zN, ztzG, baseline, alpha, kernel,
                       grad_kernel, delta, square_int_marks, n_ground_events):
    """Return the gradient of the discrete l2 loss w.r.t.
       one kernel parameters.

    .. math::
        N_T\\frac{\\partial\\mathcal{L}_G}{\\partial \\eta{ml}} =
        2\\Delta \\mu_m  \\sum_{\\tau=1}^{L} \\frac{\\partial
        \\phi_{m,l}^\\Delta[\\tau]}{\\partial \\eta{m,l}} \\Phi_l(\\tau; G)
        + 2 \\Delta \\sum_{k=1}^{p} \\sum_{\\tau=1}^{L} \\sum_{\\tau'=1}^{L}
        \\phi_{mk}^\\Delta[\\tau'] \\frac{\\partial \\phi_{m,l}^\\Delta[\\tau]}
        {\\partial \\eta{m,l}} \\Psi_{l,k}(\\tau, \\tau'; G)

    Parameters
    ----------
    zG : tensor, shape (n_dim, L)

    zN : tensor, shape (n_dim, n_dim, L)

    ztzG : tensor, shape (n_dim, n_dim, L, L)

    baseline : tensor, shape (n_dim,)
        Baseline parameter of the intensity of the Hawkes process.

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L)
        Kernel values on the discretization.

    grad_kernel : list of tensor of shape (n_dim, n_dim, L)
        Gradient values on the discretization.

    delta : float
        Step size of the discretization grid.

    n_ground_events : tensor, shape (n_dim,)
        Number of events for each dimension.

    Returns
    ----------
    grad_theta : tensor, shape (n_dim, n_dim)
    """
    n_dim, _, L = kernel.shape
    grad_theta_ = torch.zeros(n_dim, n_dim)

    if n_dim > 1:

        cst1 = 2 * alpha
        cst2 = delta * baseline.view(n_dim, 1) * cst1
        cst3 = torch.einsum('mn,mk->mkn', alpha, alpha)
        temp1 = torch.einsum('mnk,nk->mn', grad_kernel, zG)
        temp2 = torch.einsum('mnk,mnk->mn', grad_kernel, zN)
        temp3 = torch.einsum('nkuv,mnu->nkmv', ztzG, grad_kernel)
        temp4 = torch.einsum('mkv,nkmv->mknv', kernel, temp3)

        grad_theta_ = square_int_marks.unsqueeze(1) * cst2 * temp1 - cst1 * temp2 + \
            square_int_marks.unsqueeze(1) * 2 * delta * (cst3 * temp4.sum(3)).sum(1)
    else:
        for m in range(n_dim):
            cst = 2 * delta * baseline[m]
            for n in range(n_dim):
                grad_theta_[m, n] = cst * alpha[m, n] * (grad_kernel[m, n] @ zG[n]) * \
                    square_int_marks[m]
                grad_theta_[m, n] -= 2 * alpha[m, n] * (grad_kernel[m, n] @ zN[m, n])
                temp = 0
                for k in range(n_dim):
                    cst2 = alpha[m, n] * alpha[m, k]
                    temp_ = 0
                    temp_ += 2 * (kernel[m, k].view(1, L)
                                  * (ztzG[n, k] * grad_kernel[m, n].view(L, 1)).sum(0))

                    temp += cst2 * temp_.sum()

            grad_theta_[m, n] += delta * temp * square_int_marks[m]

    grad_theta = grad_theta_ / n_ground_events.sum()

    return grad_theta


class JointFaDIn(object):

    get_zN = staticmethod(get_zN_joint)

    def __init__(self, n_dim, kernel, kernel_params_init=None,
                 baseline_init=None, baseline_mask=None,
                 alpha_init=None, alpha_mask=None, moment_matching=False,
                 kernel_length=1, delta=0.01, optim='RMSprop',
                 params_optim=dict(), max_iter=2000, optimize_kernel=True,
                 precomputations=True, ztzG_approx=True, device='cpu',
                 log=False, grad_kernel=None, criterion='l2', tol=10e-5,
                 random_state=None, optimize_alpha=True):
        # Set discretization parameters
        self.delta = delta
        self.W = kernel_length
        self.L = int(self.W / delta)
        self.ztzG_approx = ztzG_approx

        # Set optimizer parameters
        self.solver = optim
        self.max_iter = max_iter
        self.log = log
        self.tol = tol

        # Set model parameters
        self.moment_matching = moment_matching
        self.n_dim = n_dim
        if baseline_init is None:
            self.baseline = torch.rand(self.n_dim)
        else:
            self.baseline = baseline_init.float()
        if baseline_mask is None:
            self.baseline_mask = torch.ones([n_dim])
        else:
            assert baseline_mask.shape == self.baseline.shape, \
                "Invalid baseline_mask shape, must be (n_dim,)"
            self.baseline_mask = baseline_mask
        self.baseline = (
            self.baseline * self.baseline_mask
        ).requires_grad_(True)
        if alpha_init is None:
            self.alpha = torch.rand(self.n_dim, self.n_dim)
        else:
            self.alpha = alpha_init.float()
        if alpha_mask is None:
            self.alpha_mask = torch.ones([self.n_dim, self.n_dim])
        else:
            assert alpha_mask.shape == self.alpha.shape, \
                "Invalid alpha_mask shape, must be (n_dim, n_dim)"
            self.alpha_mask = alpha_mask
        self.alpha = (self.alpha * self.alpha_mask).requires_grad_(True)

        if kernel_params_init is None:
            kernel_params_init = []
            if kernel == 'raised_cosine':
                temp = 0.5 * self.W * torch.rand(self.n_dim, self.n_dim)
                temp2 = 0.5 * self.W * torch.rand(self.n_dim, self.n_dim)
                kernel_params_init.append(temp)
                kernel_params_init.append(temp2)
            elif kernel == 'truncated_gaussian':
                temp = 0.25 * self.W * torch.rand(self.n_dim, self.n_dim)
                temp2 = 0.5 * self.W * torch.rand(self.n_dim, self.n_dim)
                kernel_params_init.append(temp)
                kernel_params_init.append(temp2)
            elif kernel == 'truncated_exponential':
                kernel_params_init.append(2 * torch.rand(self.n_dim,
                                                         self.n_dim))
            else:
                raise NotImplementedError(
                    'kernel initial parameters of not \
                    implemented kernel have to be given'
                )

        self.kernel_params_fixed = kernel_params_init

        self.n_kernel_params = len(kernel_params_init)

        self.kernel_model = DiscreteKernelFiniteSupport(self.delta, self.n_dim,
                                                        kernel,
                                                        self.W, 0, grad_kernel)
        self.kernel = kernel
        # Set optimizer
        if optimize_alpha:
            self.params_intens = [self.baseline, self.alpha]
        else:
            self.params_intens = [self.baseline]
            self.alpha_fixed = self.alpha.detach()

        self.optimize_kernel = optimize_kernel
        self.optimize_alpha = optimize_alpha

        if self.optimize_kernel:
            for i in range(self.n_kernel_params):
                self.params_intens.append(
                    kernel_params_init[i].float().clip(1e-4).requires_grad_(
                        True)
                )

        self.precomputations = precomputations
        # If the learning rate is not given, fix it to 1e-3
        if 'lr' not in params_optim.keys():
            params_optim['lr'] = 1e-3

        self.params_solver = params_optim
        # self.opt = optimizer(self.params_intens, params_optim, solver=optim)

        if criterion == 'll':
            self.precomputations = False

        self.criterion = criterion
        # device and seed
        if random_state is None:
            torch.manual_seed(0)
        else:
            torch.manual_seed(random_state)

        if torch.cuda.is_available() and device == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def fit(self, events, end_time):
        """Learn the parameters of the Hawkes processes on a discrete grid.

        Parameters
        ----------
        events : list of array of size number of timestamps,
         size of the list is dim.
            Each array element is a list of size two :
            [event timestamp, event mark].

        end_time : int
            The end time of the Hawkes process.

        Returns
        -------
        self : object
            Fitted parameters.
        """
        n_grid = int(1 / self.delta) * end_time + 1
        discretization = torch.linspace(0, self.W, self.L)
        events_grid, events_grid_wm = projected_grid_marked(
            events, self.delta, n_grid
        )

        sum_marks = events_grid.sum(1)
        print('sum of marks per channel:', sum_marks)
        n_ground_events = [events[i].shape[0] for i in range(len(events))]
        print('number of events is:', n_ground_events)
        n_ground_events = torch.tensor(n_ground_events)
        self.sum_marks = sum_marks
        self.n_ground_events = n_ground_events
# useless in the solver since kernel[i, j, 0] = 0.
        max_mark = torch.tensor([
            events[i][:, 1].max() for i in range(len(events))]).float()
        min_mark = torch.tensor([
            events[i][:, 1].min() for i in range(len(events))]).float()

        self.square_int_marks = 4 * ((max_mark ** 3) / 3 - (min_mark ** 3) / 3)

        if self.moment_matching:
            baseline, alpha, kernel_params_init = momentmatching_fadin(
                self, events, n_ground_events, end_time
            )
            self.baseline = baseline
            self.alpha = alpha
            # Set optimizer with moment_matching parameters
            if self.optimize_alpha:
                self.params_intens = [self.baseline, self.alpha]
            else:
                self.params_intens = [baseline]
                self.alpha_fixed = alpha.detach()

            if self.optimize_kernel:
                for i in range(2):  # range(self.n_params_kernel)
                    self.params_intens.append(
                        kernel_params_init[i].float().clip(
                            1e-4).requires_grad_(True)
                    )
        self.opt = optimizer_fadin(
            self.params_intens,
            self.params_solver,
            solver=self.solver
        )

        ####################################################
        # Precomputations
        ####################################################
        if self.precomputations:

            start = time.time()
            zG = get_zG(events_grid.double().numpy(), self.L)
            zN = self.get_zN(events_grid.double().numpy(),
                             events_grid_wm.double().numpy(),
                             self.L)

            if self.ztzG_approx:
                ztzG = get_ztzG_approx(events_grid.double().numpy(), self.L)
            else:
                ztzG = get_ztzG(events_grid.double().numpy(), self.L)

            zG = torch.tensor(zG).float()
            zN = torch.tensor(zN).float()
            ztzG = torch.tensor(ztzG).float()
            print('precomput:', time.time() - start)

        ####################################################
        # save results
        ####################################################
        self.v_loss = torch.zeros(self.max_iter)

        self.param_baseline = torch.zeros(self.max_iter + 1, self.n_dim)
        self.param_alpha = torch.zeros(self.max_iter + 1,
                                       self.n_dim, self.n_dim)
        self.param_kernel = torch.zeros(self.n_kernel_params,
                                        self.max_iter + 1,
                                        self.n_dim, self.n_dim)

        self.param_baseline[0] = self.params_intens[0].detach()

        if self.optimize_alpha:
            self.param_alpha[0] = self.params_intens[1].detach()

        # If kernel parameters are optimized
        if self.optimize_kernel:
            for i in range(self.n_kernel_params):
                self.param_kernel[i, 0] = \
                    self.params_intens[2 + i].detach()

        ####################################################
        start = time.time()
        for i in range(self.max_iter):
            print(f"Fitting model... {i/self.max_iter:6.1%}\r", end='',
                  flush=True)

            self.opt.zero_grad()
            if self.precomputations:
                if self.optimize_kernel:
                    # Update kernel
                    kernel = self.kernel_model.kernel_eval(
                        self.params_intens[2:],
                        discretization
                    )
                    grad_theta = self.kernel_model.grad_eval(
                        self.params_intens[2:],
                        discretization
                    )
                else:
                    kernel = self.kernel_model.kernel_eval(
                        self.kernel_params_fixed,
                        discretization
                    )

                if self.log:
                    self.v_loss[i] = discrete_l2_loss_precomputation(
                        zG, zN, ztzG, *self.params_intens[:2], kernel,
                        sum_marks, self.delta, end_time, n_ground_events
                    ).detach()

                if self.optimize_alpha:
                    # Update baseline
                    self.params_intens[0].grad = get_grad_baseline_joint(
                        zG,
                        self.params_intens[0],
                        self.params_intens[1],
                        kernel,
                        self.delta,
                        end_time,
                        sum_marks,
                        self.square_int_marks,
                        n_ground_events)
                else:
                    self.params_intens[0].grad = get_grad_baseline_joint(
                        zG,
                        self.params_intens[0],
                        self.alpha_fixed,
                        kernel,
                        self.delta,
                        end_time,
                        sum_marks,
                        self.square_int_marks,
                        n_ground_events)
                if self.optimize_alpha:
                    # Update alpha
                    self.params_intens[1].grad = get_grad_alpha_joint(
                        zG,
                        zN,
                        ztzG,
                        self.params_intens[0],
                        self.params_intens[1],
                        kernel,
                        self.delta,
                        self.square_int_marks,
                        n_ground_events)
                if self.optimize_kernel:
                    # Update kernel
                    for j in range(self.n_kernel_params):
                        self.params_intens[2 + j].grad = \
                            get_grad_eta_joint(zG,
                                               zN,
                                               ztzG,
                                               self.params_intens[0],
                                               self.params_intens[1],
                                               kernel,
                                               grad_theta[j],
                                               self.delta,
                                               self.square_int_marks,
                                               n_ground_events)

            else:
                intens = self.kernel_model.intensity_eval(
                    self.params_intens[0],
                    self.params_intens[1],
                    self.params_intens[2:],
                    events_grid,
                    discretization
                )
                if self.criterion == 'll':
                    loss = discrete_ll_loss_conv(intens,
                                                 events_grid,
                                                 self.delta)
                else:
                    loss = discrete_l2_loss_conv(intens,
                                                 events_grid,
                                                 self.delta)
                loss.backward()

            self.opt.step()

            # Save parameters
            self.params_intens[0].data = self.params_intens[0].data.clip(0) * \
                self.baseline_mask
            if self.optimize_alpha:
                self.params_intens[1].data = self.params_intens[1].data.clip(0) * \
                    self.alpha_mask
                self.param_alpha[i + 1] = self.params_intens[1].detach()

            self.param_baseline[i + 1] = self.params_intens[0].detach()

            # If kernel parameters are optimized
            if self.optimize_kernel:
                for j in range(self.n_kernel_params):
                    self.params_intens[2 + j].data = \
                        self.params_intens[2 + j].data.clip(1e-3)
                    self.param_kernel[j, i + 1] = \
                        self.params_intens[2 + j].detach()

            # Early stopping
            if i % 100 == 0:
                error_b = torch.abs(self.param_baseline[i + 1] -
                                    self.param_baseline[i]).max()
                error_al = torch.abs(self.param_alpha[i + 1] -
                                     self.param_alpha[i]).max()
                error_k = torch.abs(self.param_kernel[0, i + 1] -
                                    self.param_kernel[0, i]).max()

                if error_b < self.tol and error_al < self.tol and error_k < self.tol:
                    print('early stopping at iteration:', i)
                    self.param_baseline = self.param_baseline[:i + 1]
                    self.param_alpha = self.param_alpha[:i + 1]
                    for j in range(self.n_kernel_params):
                        self.param_kernel[j] = self.param_kernel[j, i + 1]
                    break
        print('iterations in ', time.time() - start)

        return self


def get_grad_baseline_noise_joint_noise2(zG, baseline, baseline_noise, alpha, kernel,
                                         delta, end_time, sum_marks, square_int_marks,
                                         n_ground_events):
    """Return the gradient of the discrete l2 loss w.r.t. the baseline.

    .. math::
        N_T\\frac{\\partial\\mathcal{L}_G}{\\partial \\mu_{m}} =
        2 T \\mu_m -  2 N_T^m + 2 \\Delta\\sum_{j=1}^{p} \\sum_{\\tau=1}^{L}
        \\phi_{mj}^\\Delta[\\tau]\\Phi_{j}(\\tau; G)

    Parameters
    ----------
    zG : tensor, shape (n_dim, L)

    baseline : tensor, shape (n_dim,)
        Baseline parameter of the intensity of the Hawkes process.

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L)
        Kernel values on the discretization.

    delta : float
        Step size of the discretization grid.

    sum_marks : tensor, shape (n_dim,)
        Sum of marks for each dimension.

    end_time : float
        The end time of the Hawkes process.

    n_ground_events : tensor, shape (n_dim,)
        Number of events for each dimension.

    Returns
    ----------
    grad_baseline: tensor, shape (dim,)
    """
    n_dim, _, _ = kernel.shape

    if n_dim > 1:
        cst1 = end_time * baseline_noise
        cst2 = 0.5 * n_ground_events.sum()
        cst3 = end_time * baseline

        dot_kernel = torch.einsum('kju,ju->kj', kernel, zG)
        dot_kernel_ = (dot_kernel * alpha).sum(1)

        grad_baseline_noise = (
            dot_kernel_ * delta + cst1 - n_ground_events + cst3) / cst2
    else:
        grad_baseline_noise_ = torch.zeros(n_dim)
        for k in range(n_dim):
            temp = 0
            for j in range(n_dim):
                temp += alpha[k, j] * (zG[j] @ kernel[k, j])
            grad_baseline_noise_[k] = delta * temp
            grad_baseline_noise_[k] += end_time * baseline_noise[k]
            grad_baseline_noise_[k] -= n_ground_events[k]
            grad_baseline_noise_[k] += end_time * baseline[k]
        grad_baseline_noise = 2 * grad_baseline_noise_ / n_ground_events.sum()

    return grad_baseline_noise


def get_grad_alpha_joint_noise2(zG, zN, ztzG, baseline, baseline_noise, alpha, kernel,
                                delta, square_int_marks, n_ground_events):
    """Return the gradient of the discrete l2 loss w.r.t. alpha.

    .. math::
        N_T\\frac{\\partial\\mathcal{L}_G}{\\partial \\alpha_{ml}} =
        2\\Delta \\mu_m  \\sum_{\\tau=1}^{L} \\frac{\\partial
        \\phi_{m,l}^\\Delta[\\tau]}{\\partial \\alpha_{m,l}} \\Phi_l(\\tau; G)
        + 2 \\Delta \\sum_{k=1}^{p} \\sum_{\\tau=1}^{L} \\sum_{\\tau'=1}^{L}
        \\phi_{mk}^\\Delta[\\tau'] \\frac{\\partial \\phi_{m,l}^\\Delta[\\tau]}
        {\\partial \\alpha_{m,l}} \\Psi_{l,k}(\\tau, \\tau'; G)

    Parameters
    ----------
    zG : tensor, shape (n_dim, L)

    zN : tensor, shape (n_dim, L)

    ztzG : tensor, shape (n_dim, n_dim, L, L)

    baseline : tensor, shape (n_dim,)
        Baseline parameter of the intensity of the Hawkes process.

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L)
        Kernel values on the discretization.

    delta : float
        Step size of the discretization grid.

    n_ground_events : tensor, shape (n_dim,)
        Number of events for each dimension.

    Returns
    ----------
    grad_alpha : tensor, shape (n_dim, n_dim)
    """
    n_dim, _, _ = kernel.shape

    if n_dim > 1:
        cst1 = delta * (baseline.view(n_dim, 1) * square_int_marks.unsqueeze(1)
                        + baseline_noise.view(n_dim, 1))

        dot_kernel = torch.einsum('njuv,knu->njkv', ztzG, kernel)
        ker_ztzg = torch.einsum('kju,njku->knj', kernel, dot_kernel)

        term1 = torch.einsum('knj,kj->kn', ker_ztzg, alpha)
        term2 = torch.einsum('knu,nu->kn', kernel, zG)
        term3 = torch.einsum('knu,knu->kn', zN, kernel)

        grad_alpha_ = square_int_marks.unsqueeze(1) * term1 * delta + \
            cst1 * term2 - term3
    else:
        grad_alpha_ = torch.zeros(n_dim, n_dim)
        for k in range(n_dim):
            dk = delta * (baseline[k] * square_int_marks[k] + baseline_noise[k])
            for n in range(n_dim):
                temp = 0
                for j in range(n_dim):
                    temp += alpha[k, j] * (torch.outer(kernel[k, n], kernel[k, j]) *
                                           ztzG[n, j]).sum()
                grad_alpha_[k, n] += delta * temp * (square_int_marks[k])
                grad_alpha_[k, n] += dk * kernel[k, n] @ zG[n]
                grad_alpha_[k, n] -= zN[k, n] @ kernel[k, n]

    grad_alpha = 2 * grad_alpha_ / n_ground_events.sum()

    return grad_alpha


def get_grad_eta_joint_noise2(zG, zN, ztzG, baseline, baseline_noise, alpha, kernel,
                              grad_kernel, delta, square_int_marks, n_ground_events):
    """Return the gradient of the discrete l2 loss w.r.t.
       one kernel parameters.

    .. math::
        N_T\\frac{\\partial\\mathcal{L}_G}{\\partial \\eta{ml}} =
        2\\Delta \\mu_m  \\sum_{\\tau=1}^{L} \\frac{\\partial
        \\phi_{m,l}^\\Delta[\\tau]}{\\partial \\eta{m,l}} \\Phi_l(\\tau; G)
        + 2 \\Delta \\sum_{k=1}^{p} \\sum_{\\tau=1}^{L} \\sum_{\\tau'=1}^{L}
        \\phi_{mk}^\\Delta[\\tau'] \\frac{\\partial \\phi_{m,l}^\\Delta[\\tau]}
        {\\partial \\eta{m,l}} \\Psi_{l,k}(\\tau, \\tau'; G)

    Parameters
    ----------
    zG : tensor, shape (n_dim, L)

    zN : tensor, shape (n_dim, n_dim, L)

    ztzG : tensor, shape (n_dim, n_dim, L, L)

    baseline : tensor, shape (n_dim,)
        Baseline parameter of the intensity of the Hawkes process.

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L)
        Kernel values on the discretization.

    grad_kernel : list of tensor of shape (n_dim, n_dim, L)
        Gradient values on the discretization.

    delta : float
        Step size of the discretization grid.

    n_ground_events : tensor, shape (n_dim,)
        Number of events for each dimension.

    Returns
    ----------
    grad_theta : tensor, shape (n_dim, n_dim)
    """
    n_dim, _, L = kernel.shape
    grad_theta_ = torch.zeros(n_dim, n_dim)

    if n_dim > 1:

        cst1 = 2 * alpha
        cst2 = delta * (baseline.view(n_dim, 1) * square_int_marks.unsqueeze(1) +
                        baseline_noise.view(n_dim, 1)) * cst1
        cst3 = torch.einsum('mn,mk->mkn', alpha, alpha)
        temp1 = torch.einsum('mnk,nk->mn', grad_kernel, zG)
        temp2 = torch.einsum('mnk,mnk->mn', grad_kernel, zN)
        temp3 = torch.einsum('nkuv,mnu->nkmv', ztzG, grad_kernel)
        temp4 = torch.einsum('mkv,nkmv->mknv', kernel, temp3)

        grad_theta_ = cst2 * temp1 - cst1 * temp2 + (
            square_int_marks.unsqueeze(1) + 1) * 2 * delta * (
                cst3 * temp4.sum(3)).sum(1)
    else:
        for m in range(n_dim):
            cst = 2 * delta * (baseline[m] * square_int_marks[m] + baseline_noise[m])
            for n in range(n_dim):
                grad_theta_[m, n] = cst * alpha[m, n] * (grad_kernel[m, n] @ zG[n])

                grad_theta_[m, n] -= 2 * alpha[m, n] * (grad_kernel[m, n] @ zN[m, n])
                temp = 0
                for k in range(n_dim):
                    cst2 = alpha[m, n] * alpha[m, k]
                    temp_ = 0
                    temp_ += 2 * (kernel[m, k].view(1, L)
                                  * (ztzG[n, k] * grad_kernel[m, n].view(L, 1)).sum(0))

                    temp += cst2 * temp_.sum()

            grad_theta_[m, n] += delta * temp * (square_int_marks[m])

    grad_theta = grad_theta_ / n_ground_events.sum()

    return grad_theta


def get_grad_baseline_joint_noise2(zG, baseline, baseline_noise, alpha, kernel,
                                   delta, end_time, sum_marks,
                                   square_int_marks, n_ground_events):
    """Return the gradient of the discrete l2 loss w.r.t. the baseline.

    .. math::
        N_T\\frac{\\partial\\mathcal{L}_G}{\\partial \\mu_{m}} =
        2 T \\mu_m -  2 N_T^m + 2 \\Delta\\sum_{j=1}^{p} \\sum_{\\tau=1}^{L}
        \\phi_{mj}^\\Delta[\\tau]\\Phi_{j}(\\tau; G)

    Parameters
    ----------
    zG : tensor, shape (n_dim, L)

    baseline : tensor, shape (n_dim,)
        Baseline parameter of the intensity of the Hawkes process.

    alpha : tensor, shape (n_dim, n_dim)
        Alpha parameter of the intensity of the Hawkes process.

    kernel : tensor, shape (n_dim, n_dim, L)
        Kernel values on the discretization.

    delta : float
        Step size of the discretization grid.

    sum_marks : tensor, shape (n_dim,)
        Sum of marks for each dimension.

    end_time : float
        The end time of the Hawkes process.

    n_ground_events : tensor, shape (n_dim,)
        Number of events for each dimension.

    Returns
    ----------
    grad_baseline: tensor, shape (dim,)
    """
    n_dim, _, _ = kernel.shape

    if n_dim > 1:
        cst1 = end_time * baseline * square_int_marks
        cst2 = 0.5 * n_ground_events.sum()
        cst3 = end_time * baseline_noise

        dot_kernel = torch.einsum('kju,ju->kj', kernel, zG)
        dot_kernel_ = (dot_kernel * alpha).sum(1) * square_int_marks

        grad_baseline = (dot_kernel_ * delta + cst1 - sum_marks + cst3) / cst2
    else:
        grad_baseline_ = torch.zeros(n_dim)
        for k in range(n_dim):
            temp = 0
            for j in range(n_dim):
                temp += alpha[k, j] * (zG[j] @ kernel[k, j])
            grad_baseline_[k] = delta * temp * square_int_marks[k]
            grad_baseline_[k] += end_time * baseline[k] * square_int_marks[k]
            grad_baseline_[k] -= 2 * sum_marks[k]  # add of joint MarkedFaDIn
            grad_baseline_[k] += end_time * baseline_noise[k]
        grad_baseline = 2 * grad_baseline_ / n_ground_events.sum()

    return grad_baseline


class JointFaDInDenoising(object):
    """Joint inference of the Hawkes process and the noise, without unmixing.

    This class is used for ablation study and should not be used in practice.
    """

    get_zN = staticmethod(get_zN_joint)

    def __init__(self, n_dim, kernel, kernel_params_init=None,
                 baseline_init=None, baseline_mask=None,
                 alpha_init=None, alpha_mask=None, moment_matching=True,
                 kernel_length=1, delta=0.01, optim='RMSprop',
                 params_optim=dict(), max_iter=2000, optimize_kernel=True,
                 precomputations=True, ztzG_approx=True, device='cpu',
                 log=False, grad_kernel=None, criterion='l2', tol=10e-5,
                 random_state=None, optimize_alpha=True):
        # Set discretization parameters
        self.delta = delta
        self.W = kernel_length
        self.L = int(self.W / delta)
        self.ztzG_approx = ztzG_approx

        # Set optimizer parameters
        self.solver = optim
        self.max_iter = max_iter
        self.log = log
        self.tol = tol

        # Set model parameters
        self.moment_matching = moment_matching
        self.n_dim = n_dim
        if baseline_init is None:
            self.baseline = torch.rand(self.n_dim)
        else:
            self.baseline = baseline_init.float()
        if baseline_mask is None:
            self.baseline_mask = torch.ones([n_dim])
        else:
            assert baseline_mask.shape == self.baseline.shape, \
                "Invalid baseline_mask shape, must be (n_dim,)"
            self.baseline_mask = baseline_mask
        self.baseline = (self.baseline * self.baseline_mask).requires_grad_(True)

        self.baseline_noise = torch.rand(self.n_dim).requires_grad_(True)

        if alpha_init is None:
            self.alpha = torch.rand(self.n_dim, self.n_dim)
        else:
            self.alpha = alpha_init.float()
        if alpha_mask is None:
            self.alpha_mask = torch.ones([self.n_dim, self.n_dim])
        else:
            assert alpha_mask.shape == self.alpha.shape, \
                "Invalid alpha_mask shape, must be (n_dim, n_dim)"
            self.alpha_mask = alpha_mask
        self.alpha = (self.alpha * self.alpha_mask).requires_grad_(True)

        if kernel_params_init is None:
            kernel_params_init = []
            if kernel == 'raised_cosine':
                temp = 0.5 * self.W * torch.rand(self.n_dim, self.n_dim)
                temp2 = 0.5 * self.W * torch.rand(self.n_dim, self.n_dim)
                kernel_params_init.append(temp)
                kernel_params_init.append(temp2)
            elif kernel == 'truncated_gaussian':
                temp = 0.25 * self.W * torch.rand(self.n_dim, self.n_dim)
                temp2 = 0.5 * self.W * torch.rand(self.n_dim, self.n_dim)
                kernel_params_init.append(temp)
                kernel_params_init.append(temp2)
            elif kernel == 'truncated_exponential':
                kernel_params_init.append(2 * torch.rand(self.n_dim,
                                                         self.n_dim))
            else:
                raise NotImplementedError(
                    'kernel initial parameters of not \
                    implemented kernel have to be given'
                )

        self.kernel_params_fixed = kernel_params_init

        self.n_kernel_params = len(kernel_params_init)

        self.kernel_model = DiscreteKernelFiniteSupport(self.delta, self.n_dim,
                                                        kernel,
                                                        self.W, 0, grad_kernel)
        self.kernel = kernel
        # Set optimizer
        if optimize_alpha:
            self.params_intens = [
                self.baseline,
                self.baseline_noise,
                self.alpha
            ]
        else:
            self.params_intens = [self.baseline, self.baseline_noise]
            self.alpha_fixed = self.alpha.detach()

        self.optimize_kernel = optimize_kernel
        self.optimize_alpha = optimize_alpha

        if self.optimize_kernel:
            for i in range(self.n_kernel_params):
                self.params_intens.append(
                    kernel_params_init[i].float().clip(1e-4).requires_grad_(
                        True
                    )
                )

        self.precomputations = precomputations
        # If the learning rate is not given, fix it to 1e-3
        if 'lr' not in params_optim.keys():
            params_optim['lr'] = 1e-3

        self.params_solver = params_optim
        self.opt = optimizer_fadin(
            self.params_intens,
            params_optim,
            solver=optim
        )

        if criterion == 'll':
            self.precomputations = False

        self.criterion = criterion
        # device and seed
        if random_state is None:
            torch.manual_seed(0)
        else:
            torch.manual_seed(random_state)

        if torch.cuda.is_available() and device == 'cuda':
            self.device = 'cuda'
        else:
            self.device = 'cpu'

    def fit(self, events, end_time):
        """Learn the parameters of the Hawkes processes on a discrete grid.

        Parameters
        ----------
        events : list of array of size number of timestamps,
         size of the list is dim.
            Each array element is a list of size two :
            [event timestamp, event mark].

        end_time : int
            The end time of the Hawkes process.

        Returns
        -------
        self : object
            Fitted parameters.
        """
        n_grid = int(1 / self.delta) * end_time + 1
        discretization = torch.linspace(0, self.W, self.L)
        events_grid, events_grid_wm = projected_grid_marked(
            events, self.delta, n_grid
        )

        sum_marks = events_grid.sum(1)
        print('sum of marks per channel:', sum_marks)
        n_ground_events = [events[i].shape[0] for i in range(len(events))]
        print('number of events is:', n_ground_events)
        n_ground_events = torch.tensor(n_ground_events)
        self.sum_marks = sum_marks
        self.n_ground_events = n_ground_events

        self.square_int_marks = torch.tensor([4/3 for _ in range(self.n_dim)])

        if self.moment_matching:
            # Smart initialization of solver parameters
            baseline, bl_noise, alpha, kernel_params_init = momentmatching_unhap(
                self, events, events_grid, n_ground_events, end_time
            )
            # Initial baseline and alpha divided by n_dim+2 instead of n_dim+1
            self.baseline = baseline
            self.bl_noise = bl_noise
            self.alpha = alpha
            # Set optimizer with moment_matching parameters
            if self.optimize_alpha:
                self.params_intens = [self.baseline, self.baseline_noise,
                                      self.alpha]
            else:
                self.params_intens = [self.baseline, self.baseline_noise]
                self.alpha_fixed = self.alpha.detach()

            if self.optimize_kernel:
                for i in range(2):  # range(self.n_params_kernel)
                    self.params_intens.append(
                        kernel_params_init[i].float().clip(1e-4).requires_grad_(True)
                    )
            self.opt = optimizer_fadin(
                self.params_intens,
                self.params_solver,
                solver=self.solver
            )

        ####################################################
        # Precomputations
        ####################################################
        if self.precomputations:

            start = time.time()
            zG = get_zG(events_grid.double().numpy(), self.L)
            zN = self.get_zN(events_grid.double().numpy(),
                             events_grid_wm.double().numpy(),
                             self.L)

            if self.ztzG_approx:
                ztzG = get_ztzG_approx(events_grid.double().numpy(), self.L)
            else:
                ztzG = get_ztzG(events_grid.double().numpy(), self.L)

            zG = torch.tensor(zG).float()
            zN = torch.tensor(zN).float()
            ztzG = torch.tensor(ztzG).float()
            print('precomput:', time.time() - start)

        ####################################################
        # save results
        ####################################################
        self.v_loss = torch.zeros(self.max_iter)

        self.param_baseline = torch.zeros(self.max_iter + 1, self.n_dim)
        self.param_baseline_noise = torch.zeros(self.max_iter + 1, self.n_dim)
        self.param_alpha = torch.zeros(self.max_iter + 1,
                                       self.n_dim, self.n_dim)
        self.param_kernel = torch.zeros(self.n_kernel_params,
                                        self.max_iter + 1,
                                        self.n_dim, self.n_dim)

        self.param_baseline[0] = self.params_intens[0].detach()
        self.param_baseline_noise[0] = self.params_intens[1].detach()

        if self.optimize_alpha:
            self.param_alpha[0] = self.params_intens[2].detach()

        # If kernel parameters are optimized
        if self.optimize_kernel:
            for i in range(self.n_kernel_params):
                self.param_kernel[i, 0] = self.params_intens[3 + i].detach()

        ####################################################
        start = time.time()
        for i in range(self.max_iter):
            print(f"Fitting model... {i/self.max_iter:6.1%}\r", end='',
                  flush=True)

            self.opt.zero_grad()
            if self.precomputations:
                if self.optimize_kernel:
                    # Update kernel
                    kernel = self.kernel_model.kernel_eval(
                        self.params_intens[3:],
                        discretization
                    )
                    grad_theta = self.kernel_model.grad_eval(
                        self.params_intens[3:],
                        discretization
                    )
                else:
                    kernel = self.kernel_model.kernel_eval(
                        self.kernel_params_fixed,
                        discretization
                    )

                if self.log:
                    #  Not done
                    self.v_loss[i] = \
                        discrete_l2_loss_precomputation(
                            zG, zN, ztzG,
                            self.params_intens[0],
                            self.params_intens[1],
                            kernel, sum_marks,
                            self.delta,
                            end_time,
                            n_ground_events
                        ).detach()

                self.params_intens[0].grad = get_grad_baseline_joint_noise2(
                    zG,
                    self.params_intens[0],
                    self.params_intens[1],
                    self.params_intens[2],
                    kernel,
                    self.delta,
                    end_time,
                    sum_marks,
                    self.square_int_marks,
                    n_ground_events)

                self.params_intens[1].grad = get_grad_baseline_noise_joint_noise2(
                    zG,
                    self.params_intens[0],
                    self.params_intens[1],
                    self.params_intens[2],
                    kernel,
                    self.delta,
                    end_time,
                    sum_marks,
                    self.square_int_marks,
                    n_ground_events)

                if self.optimize_alpha:
                    # Update alpha
                    self.params_intens[2].grad = get_grad_alpha_joint_noise2(
                        zG,
                        zN,
                        ztzG,
                        self.params_intens[0],
                        self.params_intens[1],
                        self.params_intens[2],
                        kernel,
                        self.delta,
                        self.square_int_marks,
                        n_ground_events)
                if self.optimize_kernel:
                    # Update kernel
                    for j in range(self.n_kernel_params):
                        self.params_intens[3 + j].grad = \
                            get_grad_eta_joint_noise2(zG,
                                                      zN,
                                                      ztzG,
                                                      self.params_intens[0],
                                                      self.params_intens[1],
                                                      self.params_intens[2],
                                                      kernel,
                                                      grad_theta[j],
                                                      self.delta,
                                                      self.square_int_marks,
                                                      n_ground_events)

            else:
                #  Not done
                intens = self.kernel_model.intensity_eval(
                    self.params_intens[0],
                    self.params_intens[1],
                    self.params_intens[2:],
                    events_grid,
                    discretization
                )

                #   Not done
                if self.criterion == 'll':
                    loss = discrete_ll_loss_conv(intens,
                                                 events_grid,
                                                 self.delta)
                else:
                    loss = discrete_l2_loss_conv(intens,
                                                 events_grid,
                                                 self.delta)
                loss.backward()

            self.opt.step()

            # Save parameters
            self.params_intens[0].data = self.params_intens[0].data.clip(0) * \
                self.baseline_mask
            self.params_intens[1].data = self.params_intens[1].data.clip(0)
            if self.optimize_alpha:
                self.params_intens[2].data = self.params_intens[2].data.clip(0) * \
                    self.alpha_mask
                self.param_alpha[i + 1] = self.params_intens[2].detach()

            self.param_baseline[i + 1] = self.params_intens[0].detach()
            self.param_baseline_noise[i + 1] = self.params_intens[1].detach()

            # If kernel parameters are optimized
            if self.optimize_kernel:
                for j in range(self.n_kernel_params):
                    self.params_intens[3 + j].data = \
                        self.params_intens[3 + j].data.clip(1e-3)
                    self.param_kernel[j, i + 1] = self.params_intens[3 + j].detach()

            # Early stopping
            if i % 100 == 0:
                error_b = torch.abs(self.param_baseline[i + 1] -
                                    self.param_baseline[i]).max()
                error_al = torch.abs(self.param_alpha[i + 1] -
                                     self.param_alpha[i]).max()
                error_k = torch.abs(self.param_kernel[0, i + 1] -
                                    self.param_kernel[0, i]).max()

                if error_b < self.tol and error_al < self.tol and error_k < self.tol:
                    print('early stopping at iteration:', i)
                    self.param_baseline = self.param_baseline[:i + 1]
                    self.param_baseline_noise = self.param_baseline_noise[:i + 1]
                    self.param_alpha = self.param_alpha[:i + 1]
                    for j in range(self.n_kernel_params):
                        self.param_kernel[j] = self.param_kernel[j, i + 1]
                    break
        print('iterations in ', time.time() - start)

        return self


# %% Define utilitary functions
def simulate_data(baseline, baseline_noise, alpha, end_time, seed=0):
    n_dim = len(baseline)

    def identity(x, **param):
        return x

    def linear_zero_one(x, **params):
        temp = 2 * x
        mask = x > 1
        temp[mask] = 0.
        return temp

    def truncated_gaussian(x, **params):
        rc = DiscreteKernelFiniteSupport(delta=0.01, n_dim=1,
                                         kernel='truncated_gaussian')
        mu = params['mu']
        sigma = params['sigma']
        kernel_values = rc.kernel_eval(
            [torch.Tensor(mu), torch.Tensor(sigma)], torch.tensor(x))

        return kernel_values.double().numpy()

    marks_kernel = identity
    marks_density = linear_zero_one
    time_kernel = truncated_gaussian

    params_marks_density = dict()
    # params_marks_density = dict(scale=1)
    params_marks_kernel = dict(slope=1.2)
    params_time_kernel = dict(mu=mu, sigma=sigma)

    marked_events, _ = simu_marked_hawkes_cluster(
        end_time, baseline, alpha, time_kernel, marks_kernel, marks_density,
        params_marks_kernel=params_marks_kernel,
        params_marks_density=params_marks_density,
        time_kernel_length=None, marks_kernel_length=None,
        params_time_kernel=params_time_kernel, random_state=seed)

    noisy_events_ = simu_multi_poisson(end_time, [baseline_noise])

    random_marks = [
        np.random.rand(noisy_events_[i].shape[0]) for i in range(n_dim)]
    noisy_events = [
        np.concatenate((noisy_events_[i].reshape(-1, 1),
                        random_marks[i].reshape(-1, 1)), axis=1) for i in range(n_dim)]

    events = [
        np.concatenate(
            (noisy_events[i], marked_events[i]), axis=0) for i in range(n_dim)]

    events_cat = [events[i][events[i][:, 0].argsort()] for i in range(n_dim)]
    # put the mark to one to test the impact of the marks
    # events_cat[0][:, 1] = 1.

    return events_cat


def run_experiment(baseline, baseline_noise, alpha, end_time, delta, seed):

    events = simulate_data(baseline, baseline_noise, alpha,
                           end_time=end_time, seed=seed)

    max_iter = 10000
    start = time.time()
    solver = UNHaP(
        n_dim=1,
        kernel="truncated_gaussian",
        kernel_length=1.,
        delta=delta, optim="RMSprop",
        params_optim={'lr': 1e-3},
        max_iter=max_iter,
        batch_rho=200
    )

    solver.fit(events, end_time)

    comp_time_mix = time.time() - start
    results = dict(param_baseline_mix=solver.param_baseline_[-10:].mean().item(),
                   param_baseline_noise_mix=solver.param_baseline_noise_[
                       -10:].mean().item(),
                   param_alpha_mix=solver.param_alpha_[-10:].mean().item(),
                   param_mu_mix=solver.param_kernel_[0][-10:].mean().item(),
                   param_sigma_mix=solver.param_kernel_[1][-10:].mean().item())

    start = time.time()
    solver = JointFaDIn(
        n_dim=1,
        kernel="truncated_gaussian",
        kernel_length=1.,
        delta=delta, optim="RMSprop",
        params_optim={'lr': 1e-3},
        max_iter=max_iter, criterion='l2',
        optimize_kernel=True,
        optimize_alpha=True
    )

    solver.fit(events, end_time)

    comp_time_joint = time.time() - start
    results["param_baseline_joint"] = solver.param_baseline[-10:].mean().item()
    results["param_alpha_joint"] = solver.param_alpha[-10:].mean().item()
    results["param_mu_joint"] = solver.param_kernel[0][-10:].mean().item()
    results["param_sigma_joint"] = solver.param_kernel[1][-10:].mean().item()

    start = time.time()
    solver = JointFaDInDenoising(
        n_dim=1,
        kernel="truncated_gaussian",
        kernel_length=1.,
        delta=delta, optim="RMSprop",
        params_optim={'lr': 1e-3},
        max_iter=max_iter, criterion='l2',
        optimize_kernel=True,
        optimize_alpha=True
    )

    solver.fit(events, end_time)
    comp_time_denoise = time.time() - start

    results["param_baseline_denoise"] = solver.param_baseline[-10:].mean().item()
    results["param_alpha_denoise"] = solver.param_alpha[-10:].mean().item()
    results["param_mu_denoise"] = solver.param_kernel[0][-10:].mean().item()
    results["param_sigma_denoise"] = solver.param_kernel[1][-10:].mean().item()

    results["seed"] = seed
    results["end_time"] = end_time
    results["delta"] = delta
    results["noise"] = baseline_noise.item()
    results["comp_time_mix"] = comp_time_mix
    results["comp_time_joint"] = comp_time_joint
    results["comp_time_denoise"] = comp_time_denoise

    return results


# Parameters
setting = 'high'  # high
baseline = np.array([.8])
mu = np.array([[0.5]])
sigma = np.array([[0.1]])

# low structure
if setting == 'low':
    alpha = np.array([[1.3]])
else:
    # high structure
    alpha = np.array([[1.45]])


delta = 0.01
end_time_list = [100, 10_00, 100_00]
baseline_noise_list = np.linspace(0.1, 1.5, 10)
seeds = np.arange(100)

n_jobs = 10
all_results = Parallel(n_jobs=n_jobs, verbose=10)(
    delayed(run_experiment)(baseline, baseline_noise,
                            alpha, end_time,
                            delta, seed=seed)
    for end_time, baseline_noise, seed in itertools.product(
        end_time_list, baseline_noise_list, seeds
    )
)

# save results
df = pd.DataFrame(all_results)
true_param = {'baseline': baseline.item(), 'alpha': alpha.item(),
              'mu': mu.item(), 'sigma': sigma.item()}
for param, value in true_param.items():
    df[param] = value


def compute_norm2_error_joint(s):
    return np.sqrt(np.array([(s[param] - s[f'param_{param}_joint'])**2
                            for param in ['baseline', 'alpha', 'mu', 'sigma']]).sum())


def compute_norm2_error_mix(s):
    return np.sqrt(np.array([(s[param] - s[f'param_{param}_mix'])**2
                            for param in ['baseline', 'alpha', 'mu', 'sigma']]).sum())


def compute_norm2_error_denoise(s):
    return np.sqrt(np.array([(s[param] - s[f'param_{param}_denoise'])**2
                            for param in ['baseline', 'alpha', 'mu', 'sigma']]).sum())


df['err_norm2_jointfadin'] = df.apply(
    lambda x: compute_norm2_error_joint(x), axis=1)
df['err_norm2_mixture'] = df.apply(
    lambda x: compute_norm2_error_mix(x), axis=1)
df['err_norm2_denoise'] = df.apply(
    lambda x: compute_norm2_error_denoise(x), axis=1)


df.to_csv(f'results/error_denoising_infer_{setting}.csv', index=False)
