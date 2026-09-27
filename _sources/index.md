# Numerical Linear Algebra: Theory, Algorithms, and Computation

These are the lecture notes and computational laboratories for a
one-semester graduate course in numerical linear algebra. The course develops
the algorithms at the core of scientific computing, namely the solution of
linear systems, least-squares problems, and eigenvalue and singular value
problems. For each algorithm it treats three questions together: what the
mathematics guarantees, what happens in floating-point arithmetic, and how the
method behaves in practice on real data.

Each chapter has two parts that are meant to be used together.

- **The notes** develop the theory: definitions, theorems with proofs,
  derivations of algorithms, their cost and stability, worked examples, and
  exercises. The notes are also available as a PDF.
- **The lab notebook** is an executable Jupyter notebook. It implements the
  algorithms of the chapter from scratch where that is instructive, checks the
  implementations against NumPy and SciPy, and runs the experiments that
  confirm (or delimit) the theory. Several chapters add a case study on a
  public dataset.

Every notebook page on this site has a download button and launch buttons for
Binder and Google Colab, so you can run and modify the code yourself.

## Where to start

- {doc}`front/syllabus` describes the scope of the course, the prerequisites
  and a suggested schedule.
- {doc}`front/notation` collects the notation used throughout.
- {doc}`front/setup` explains how to install the software and run the
  notebooks locally.

## Course map

| Part | Chapters |
|---|---|
| I. Foundations | 1. Matrix computations, orthogonality, and norms; 2. The singular value decomposition |
| II. Floating point, conditioning, and stability | 3. Floating-point arithmetic; 4. Conditioning and stability |
| III. Linear systems | 5. Gaussian elimination and LU factorization; 6. Cholesky factorization and iterative refinement |
| IV. Orthogonalization and least squares | 7. QR factorization; 8. Linear least-squares problems |
| V. Eigenvalue problems | 9. Eigenvalues: theory and vector iterations; 10. The QR algorithm |
| VI. Large-scale methods | 11. Sparse matrices and Krylov subspace methods; 12. Randomized numerical linear algebra |

## License and citation

The notes, notebooks and code are released under the license stated in the
repository. If you use these materials, please cite the repository.
