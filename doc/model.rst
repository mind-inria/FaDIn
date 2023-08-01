Model description
==================

FaDIn is a library to perform Multivariate Hawkes Processes Inference. 
It is a general framework that allows to estimate the intensity 
function of Hawkes processes with any parametric kernels having 
finite support. It is based on a gradient-based optimization of
the discrete :math:`\ell_2` loss.


**Contents**

* :ref:`intro`
* :ref:`discrete`
* :ref:`fadin`
* :ref:`kernels`

.. _intro:


1. Hawkes process
=================

A Hawkes process is a :math:`p`-dimensional counting process
:math:`N(t) = (N_1(t) \cdots N_p(t))`, where each coordinate is a counting
process :math:`N_i(t) = \sum_{n \geq 1} \mathbf 1_{t_{n}^i \leq t}`, with
:math:`t_{1}^i, t_{2}^i, \ldots` being the ticks or timestamps observed on
component :math:`i`. The intensity function of :math:`N` is given by

.. math::

    \lambda_i(t) = \mu_i + \sum_{j=1}^p \sum_{n \ : \ t_{n}^j < t} \phi_{ij}(t - t_{n}^j)

for :math:`i = 1, \ldots, p`. Elements :math:`\mu_i` are called the baseline parameters. A process :math:`j` induces intensity on another 
process :math:`i`, with an impact encoded by the *kernel* function 
:math:`\phi_{ij}`. In the the above formula,

* :math:`p` is the number of processes
* :math:`\mu_i` are the baseline intensities
* :math:`\phi_{ij}` are the kernels.

Note that different choices for the shape of the kernels correspond to different
models. Usually limited to exponential, neural networks based or non-parametric kernels;
FaDIn allows the use of any parametric kernels in an efficient way. Some pre-implemented kernels
are given in :ref:`kernels`.

.. _discrete:


2. Discretization and finite-support kernels
============================================

Define :math:`p` sets of timestamps :math:`\mathscr{F}_T^i = \left\{t_1^i, \ldots, t_{N_i}^i  \right\}`  on :math:`[0, T]`.
Given a discrete grid :math:`\mathcal{G}` of size :math:`G` such that :math:`\mathcal{G}=\left\{0, \Delta, \ldots, G \Delta \right\}`, we obtain
the discretized version of :math:`\mathscr{F}_T^i` by projecting these sets of timestamps onto the grid :math:`\mathcal{G}`. These projected timestamps are denoted by :math:`\tilde{\mathscr{F}}_T^i = \left\{\tilde{t}_1^i, \ldots, \tilde{t}_{N_i}^i \right\}`. Therefore, the discrete version of the intensity function defined in :ref:`intro`:


.. math::
		\tilde{\lambda}_i[s] = \mu_i + \sum_{j=1}^{p} \underbrace{\sum_{\tau=1}^L \phi_{ij}^\Delta[\tau] z_j[s-\tau]}_{(\phi_{ij}^\Delta * z_j)[s]},
    \quad s\in [|0, G |],

where :math:`\phi_{ij}^\Delta[\tau]=\phi_{ij}(\tau \Delta)` is the discrete kernel and :math:`L=\lfloor \frac{W}{\Delta}\rfloor` is the discrete size of the kernels :math:`\phi_{ij}` of length :math:`W`.

.. _fadin:


3. FaDIn
========

FaDIn is a :math:`\ell_2` loss based solver to infer baseline parameters :math:`\mu_i` and parameters :math:`\eta_{ij}`  of any parametric kernels :math:`\phi_{ij}:=\phi_{ij}^{\eta_{ij}}`. It is based on the discretization and finite-support kernels introduced in :ref:`discrete`. Precisely,
the discretized :math:`\ell_2` loss minimized by FaDIn is:

.. math::

  		\frac{1}{N_T}\sum_{i=1}^{p}  \left(\Delta\sum_{s\in [|0, G|]}  \left(\tilde{\lambda}_{i}[s]\right)^2 
            - 2\sum_{\tilde{t}_{n}^{i} \in \tilde{\mathscr{F}}_{T}^{i}}  \tilde{\lambda}_{i} \left[\frac{\tilde{t}_n^{i}}{\Delta}\right] \right),

where kernels involved in :math:`\tilde{\lambda}_{i}` are parametric kernels.

.. _kernels:


4. Kernels
==========

In this package, the three following kernels are implemented:

* **Raised Cosine kernel** 

.. math::
		\phi_{ij}(\cdot) = \alpha_{i,j} \left[{1 + \cos \left(\frac{\cdot - u_{i,j}}{\sigma_{i,j}}\pi - \pi \right)} \right] , \quad (i,j)\in \{1,\ldots, p\}^2.


The parameters estimated are then the triplets :math:`\eta_{ij}=\left(\alpha_{ij}, u_{ij}, \sigma_{ij}\right)`.

* **Truncated Gaussian kernel** 

.. math::
		\phi_{ij} (\cdot)= \frac{ \alpha_{ij}}{\sigma_{ij}} \frac{f\left(\frac{\cdot-m_{ij}}{\sigma_{ij}}\right)}{F\left(\frac{W-m_{ij}}{\sigma_{ij}}\right)-F\left(\frac{-m_{ij}}{\sigma_{ij}}\right)} \mathbb{I} \left\{0\leq \cdot \leq W \right\}, \quad (i,j)\in \{1,\ldots, p\}^2,


where  :math:`F` is the cdf of the Gaussian distribution. The parameters estimated are then the triplets  :math:`\eta_{ij}=\left(\alpha_{ij}, m_{ij}, \sigma_{ij}\right)`.

* **Truncated Exponential kernel** 

.. math::
		\phi_{ij} (\cdot)=  \frac{\beta_{ij} \exp(-\beta_{ij}~ \cdot)}{H\left(W\right)-H\left(0\right)} \mathbb{I} \left\{0\leq \cdot \leq W \right\}, \quad (i,j)\in \{1,\ldots, p\}^2,

where  :math:`H` is the cdf of the exponential distribution. The parameters estimated are then the doublets :math:`\eta_{ij}=\left( \alpha_{ij}, \beta_{ij} \right)`.

