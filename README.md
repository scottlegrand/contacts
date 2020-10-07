# Contact descriptor for protein-ligand interactions

This code implements a GPU-accelerated variant of a contact calculation between
a protein and a ligand. It takes a set of coordinates and atom types
as an input, and counts the number of contacts that fall in specific bins.
The output format is the same as the one generated by the RFScore2
descriptor in [ODDT](https://github.com/oddt/oddt).

A python (cupy) interface to featureize N ligands against the same protein
is provided.

Files:

- contacts.cu: prototype implementation of the kernel for a single ligand