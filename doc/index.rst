FaDIn 
================
================


Official library for using FaDIn [1]. While exponential kernels are more data efficient and relevant for certain applications where events immediately trigger more events, they are ill-suited for applications where latencies need to be estimated, such as in neuroscience. This work aims to offer an efficient solution to TPP inference using general parametric kernels with finite support. The developed solution consists of a fast L2 gradient-based solver lever- aging a discretized version of the events.


Installation
------------

To install ``FaDIn``, do::

   $ pip install fadin

If you do not have admin privileges on the computer, use the ``--user`` flag
with `pip`. To upgrade, use the ``--upgrade`` flag provided by `pip`.


Dependencies
------------

These are the dependencies to use FaDIn:

* scipy
* numpy (>=1.2)
* matplotlib (>=3)
* torch (>= 1.12.1)
* numba (0.55.2)


Cite
----

   [1] Guillaume Staerman, Cédric Allain, Alexandre Gramfort, Thomas Moreau. FaDIn: Fast Discretized Inference for Hawkes Processes with General Parametric Kernels. Preprint (2022). https://arxiv.org/abs/2210.04635.


Contents
--------

.. toctree::
        :maxdepth: 2

        model
        examples
        api