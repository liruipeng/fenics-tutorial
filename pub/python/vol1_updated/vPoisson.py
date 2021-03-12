"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  -div(kappa*grad(u)) = f
            u = u_D  on the boundary
"""

from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy
from scipy.sparse import csr_matrix
from scipy.io import mmwrite

import pdb

def printCoords(V):
    # note that the ordering of dof can be different from mesh vertices given by
    # coordinates = mesh.coordinates()
    # the mapping between can be got by
    # v2d = vertex_to_dof_map(V)
    coords = V.tabulate_dof_coordinates()
    ndofs  = V.dim()
    nnodes = coords.shape[0]
    dim = coords.shape[1]
    print("Printing coordinates for mesh with",ndofs,"nodes and dimension",dim)
    header  = "%24d %24d" %(ndofs, dim)
    if(dim == 2):
        numpy.savetxt("mat.coords", coords, header=header, comments='', fmt="%24.15e %24.15e")
    elif(dim == 3):
        numpy.savetxt("mat.coords", coords, header=header, comments='', fmt="%24.15e %24.15e %24.15e")

def vPoisson():
    # Create mesh and define function space
    mesh = UnitSquareMesh(32, 32)
    V = FunctionSpace(mesh, 'P', 1)

    #pdb.set_trace()
    #printCoords(V)

    # Define boundary condition
    # u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
    # u_D = Constant(1)
    a = 4*pi
    b = 4*pi
    u_D = Expression('sin(a*x[0]-pi/2)*sin(b*x[1]-pi/2)', degree=6, a=a, b=b)
    kappa = Expression('sin(a*x[0]+b*x[1])', degree=6, a=a, b=b)
    f = Expression('(a*a+b*b)*sin(a*x[0]-pi/2)*sin(b*x[1]-pi/2)', degree=6, a=a, b=b)

    #plot(kappa, mesh=mesh)
    c = plot(kappa, mesh=mesh)
    plt.colorbar(c)
    plt.show()

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u_D, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = dot(grad(u), grad(v))*dx
    L = f*v*dx

    A, b = assemble_system(a, L, bc)

    # Assemble linear system
    #A = assemble(a)
    #b = assemble(L)

    # Apply boundary conditions
    #bc.apply(A, b)

    Acsr = csr_matrix(A.array())

    mmwrite('A.mtx',Acsr,symmetry='general')
    numpy.savetxt("A.txt", A.array())

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    # Plot solution and mesh
    plot(u)
    #plot(mesh)

    # Save solution to file in VTK format
    #vtkfile = File('poisson/solution.pvd')
    #vtkfile << u

    A_array = A.array()
    u_array = u.vector().get_local()
    b_array = b.get_local()
    r_array = b_array - A_array @ u_array

    # Print errors
    print('error_max =', numpy.linalg.norm(r_array))

    # Hold plot
    plt.show()

if __name__ == '__main__':
    vPoisson()
