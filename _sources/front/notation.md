(front-notation)=
# Notation

The same notation is used in the notes, the notebooks and the exercises.

## Spaces, matrices and vectors

| Symbol | Meaning |
|---|---|
| $\R^n$, $\R^{m\times n}$ | real column vectors of length $n$; real matrices with $m$ rows and $n$ columns |
| $\C^n$, $\C^{m\times n}$ | the complex analogues |
| $A$, $x$, $\alpha$ | matrices are uppercase, vectors lowercase, scalars lowercase or Greek |
| $a_{ij}$, $a_j$ | the $(i,j)$ entry and the $j$th column of $A$, so $A = [a_1,\dots,a_n]$ |
| $A^T$, $A^*$ | transpose; conjugate transpose |
| $I$, $I_n$, $e_j$ | identity matrix (of order $n$); $j$th column of the identity |
| $\range(A)$, $\nullsp(A)$, $\rank(A)$ | range (column space), null space and rank |
| $\diag(d_1,\dots,d_n)$ | diagonal matrix with the given diagonal entries |
| $\trace(A)$, $\det(A)$ | trace and determinant |
| $\Span\{v_1,\dots,v_k\}$ | the set of linear combinations of $v_1,\dots,v_k$ |
| $x^{(k)}$ | the $k$th iterate of an iterative method (a superscript in parentheses, never a subscript) |

## Norms, singular values and eigenvalues

| Symbol | Meaning |
|---|---|
| $\lVert x\rVert_p$ | vector $p$-norm, $1\le p\le\infty$ |
| $\lVert A\rVert_p$, $\lVert A\rVert_F$ | induced matrix $p$-norm; Frobenius norm |
| $\sigma_1\ge\dots\ge\sigma_p\ge0$ | singular values of $A\in\R^{m\times n}$, $p = \min(m,n)$ |
| $\lambda_i$, $\rho(A)$ | eigenvalues; spectral radius $\max_i\lvert\lambda_i\rvert$ |
| $\kappa(A)$, $\kappa_2(A)$ | condition number $\lVert A\rVert\,\lVert A^{-1}\rVert$; in the 2-norm, $\sigma_1/\sigma_n$ |
| $A^+$ | Moore--Penrose pseudoinverse |

## Floating-point arithmetic

| Symbol | Meaning |
|---|---|
| $\fl(x)$ | the floating-point number nearest to $x$ (round to nearest) |
| $u$ | unit roundoff, $u = 2^{-53}\approx1.11\times10^{-16}$ in IEEE double precision |
| $\epsmach$ | machine epsilon, the distance from $1$ to the next larger float, $2^{-52} = 2u$ |
| $\gamma_n$ | $nu/(1-nu)$, defined when $nu<1$ |

The standard model of floating-point arithmetic is
$\fl(x\mathbin{\mathrm{op}}y) = (x\mathbin{\mathrm{op}}y)(1+\delta)$ with
$\lvert\delta\rvert\le u$. Some books, including {cite:t}`trefethen1997`, call
the unit roundoff "machine epsilon" and write it
$\epsilon_{\text{machine}}$. When we quote such results we translate them to
$u$. NumPy's `np.finfo(float).eps` is $\epsmach = 2u$.

## Cost

A *flop* is one floating-point addition, subtraction, multiplication or
division. Operation counts are given by their leading term, for example
$\tfrac23n^3$ flops for LU factorization. We write $f(n) = \bigO(g(n))$ when
$\lvert f(n)\rvert\le Cg(n)$ for some constant $C$ and all sufficiently large
$n$ (or, in error bounds, all sufficiently small $u$).

## Cross-references

Definitions, theorems, algorithms and exercises are numbered by chapter:
Theorem 5.3 is the third theorem of Chapter 5. Equations are numbered in the
same way and cited in parentheses, as in (5.2).
