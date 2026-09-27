(front-syllabus)=
# About the course

## Scope

The course is an introduction to numerical linear algebra at the graduate
level. It is written for students in applied mathematics, mathematics,
statistics, data science, engineering and the computational sciences. The
central questions are how to compute the solutions of the standard problems
of linear algebra, how much the answers can be trusted when every operation is
rounded, and how the algorithms scale to large problems.

The material is organized in six parts. Part I develops the language of the
subject: matrix products and their cost, orthogonal transformations, norms,
and the singular value decomposition. Part II explains floating-point
arithmetic and the two ideas that organize all of error analysis,
conditioning (a property of the problem) and stability (a property of the
algorithm). Parts III and IV treat linear systems and least-squares problems
through the LU, Cholesky and QR factorizations. Part V treats the eigenvalue
problem, from the power method to the shifted QR algorithm that underlies
every modern eigenvalue solver. Part VI turns to large problems: sparse
matrices, Krylov subspace methods such as conjugate gradients and GMRES,
preconditioning, and randomized algorithms for low-rank approximation and
least squares.

## Prerequisites

Students should be comfortable with undergraduate linear algebra (vector
spaces, linear maps, rank, determinants, eigenvalues, the spectral theorem for
symmetric matrices) and with basic real analysis at the level of limits and
continuity. Programming experience in Python with NumPy is assumed at the
level of writing functions and loops and manipulating arrays. No prior course
in numerical analysis is required.

## How the materials fit together

Each chapter consists of the notes and a lab notebook, and some chapters add
a case study. The notes are self-contained, and every theorem is either proved,
proved in outline with the outline labeled as such, or cited to a specific
source. The notebooks repeat only as much theory as they need, and they cite
the numbered results of the notes. Exercises at the end of each chapter range
from conceptual questions to small projects. Solutions to about a third of
them are included in the notes; the remaining solutions are in the
instructor's edition.

Two conventions run through the whole course. First, every numerical claim
in the notes has been computed, and the lab lets you reproduce it. Second,
run times are always presented as observations on a particular machine; the
machine-independent measures of cost are flop counts and the amount of data
moved.

## Suggested schedule

Covering every section of the notes at a comfortable pace takes about 36
lectures of 75 minutes. A 15-week semester with two lectures per week has
about 30, so a semester course omits or shortens some material; the
instructor's edition marks what can be left out of each chapter. The
schedule below is one such 30-lecture plan.

| Weeks | Lectures | Chapter |
|---|---|---|
| 1 | 2 | 1. Matrix computations, orthogonality, and norms |
| 2 | 2 | 2. The singular value decomposition |
| 3 | 2 | 3. Floating-point arithmetic |
| 4 | 2 | 4. Conditioning and stability |
| 5--6 | 3 | 5. Gaussian elimination and LU factorization |
| 6--7 | 2 | 6. Cholesky factorization and iterative refinement |
| 7--8 | 3 | 7. QR factorization |
| 9--10 | 3 | 8. Linear least-squares problems |
| 10--11 | 3 | 9. Eigenvalues: theory and vector iterations |
| 12--13 | 3 | 10. The QR algorithm |
| 13--14 | 3 | 11. Sparse matrices and Krylov subspace methods |
| 15 | 2 | 12. Randomized numerical linear algebra |

## Textbooks

The notes follow no single textbook, but they draw on and cite a small set of
standard references. {cite:t}`trefethen1997` is the closest in spirit and
level. {cite:t}`golub2013` is the comprehensive reference for algorithms,
{cite:t}`higham2002` for rounding error analysis, and {cite:t}`demmel1997` for
an applied treatment with attention to software. {cite:t}`saad2003` covers the
iterative methods of Part VI in depth.
