# GP_2021_2

Code to reproduce the figures and results in the preprint ["Analytical solutions of the chemical master equation with bursty production and isomerization reactions"](https://www.biorxiv.org/content/10.1101/2021.03.24.436847v2) by Gennady Gorin and Lior Pachter. 

``dag_cme_burst.py`` implements the DAG generation, stochastic matrix construction, simulation, and analytical solution routines. 

``gg211028_dag_2.ipynb`` demonstrates that the algorithm concords with simulations, and benchmarks the algorithm's runtime.

``gg211030_fltseq_5.ipynb`` applies the algorithm to an FLT-seq long-read sequencing dataset and validates the consistency of theoretical correlation constraints on inter-gene and intra-gene correlation matrices.
