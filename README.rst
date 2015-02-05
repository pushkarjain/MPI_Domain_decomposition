README
======

About
-----

C++ code written with MPI to analyze the scalability of matrix-vector product based on row wise and block wise domain decomposition. Solves the Poissons equation using finite difference scheme.

Generate binary
---------------

Based on the domain decomposition type, the binary **out** can be generated in the following way - 
 
Domain decomposition by blocks::
    
    make block

Domain decomposition by rows::
    
    make row

Usage
-----

The code can be executed using::

    ./out
