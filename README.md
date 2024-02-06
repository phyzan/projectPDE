This library focuses on solving linear partial differential equations (linear PDE's) of second order, in any dimensions (although more than 2 requires significant computer power).
Time-dependent and non time dependent problems are both included.
The user can input Dirichlet, Neumann, or Robin boundary conditions at the boundaries of the grid, or even in curves or single points in the interior of the grid (like in the videos in
the repository folder).
1D and 2D solutions can be plotted or animated (the latter for time dependent problems).
The library also supports handling scalar and vector fields (which can be generated from solutions of PDE's), such as computing gradient, divergence and curl.
All of the above are only supported in cartesian coordinates, however the user can construct their own linear operator manually, such
as the Laplace operator in polar coordinates, and solve the Laplace equation in those coordinates.
The work can be generalized using a metric tensor, for switching to curvilinear coordinates.

The solutions are computed using finite differences, in a uniform grid. The user can specify the accuracy of the finite differences.
Time-dependent problems of first order in time (such as the diffusion equation) use the Crank-Nicolson method, while second order problems in time
use an explicit method.
