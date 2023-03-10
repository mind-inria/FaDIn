Model description
==================

FaDIn is a library to perform Multivariate Hawkes Processes Inference. 
It is a general framework that allows to inference the intensity 
function of Hawkes processes with any parametric kernels having 
finite support. It is based on a gradient-based optimization of
the discrete :math:`\ell_2` loss.


**Contents**

* :ref:`intro`
* :ref:`fadin`

.. _intro:


1. Introduction
===============

A Hawkes process is a :math:`p`-dimensional counting process
:math:`N(t) = (N_1(t) \cdots N_p(t))`, where each coordinate is a counting
process :math:`N_i(t) = \sum_{k \geq 1} \mathbf 1_{t_{i, k} \leq t}`, with
:math:`t_{i, 1}, t_{i, 2}, \ldots` being the ticks or timestamps observed on
component :math:`i`. The intensity of :math:`N` is given by

.. math::

    \lambda_i(t) = \mu_i + \sum_{j=1}^p \sum_{k \ : \ t_{j, k} < t} \phi_{ij}(t - t_{j, k})

for :math:`i = 1, \ldots, p`. Such an intensity induces a cluster effect, namely activity one a node
:math:`j` induces intensity on another node :math:`i`, with an impact encoded
by the *kernel* function :math:`\phi_{ij}`. In the the above formula, we have

* :math:`p` is the number of processes
* :math:`\mu_i` are the baseline intensities
* :math:`\phi_{ij}` are the kernels.

Note that different choices for the shape of the kernels correspond to different
models. Usually limited to exponential, neural networks based or non-parametric kernels;
FaDIn allows the use of any parametric kernels in an efficient way. 



2. FaDIn
===============


Given a discrete grid :math:`\mathcal{G}`

.. math::
  		\frac{1}{N_T}\sum_{i=1}^{p}  \left(\Delta\sum_{s\in [0, G] \right) \left(\tilde{\lambda}_{i}[s]}^2
            - 2\sum_{\tilde{t}_n^i \in \widetilde{\mathscr{F}}_T^i}\tilde{\lambda}_{i} \left[\frac{\tilde{t}_n^{i}}{\Delta}}\right],

where 