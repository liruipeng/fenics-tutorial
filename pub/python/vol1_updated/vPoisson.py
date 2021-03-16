"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  -div(kappa*grad(u)) = f
            u = u_D  on the boundary
"""

from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy
from numpy import linalg as LA
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

def surfaceplot(nx, ny, u):
    V = u.function_space()
    v2d = vertex_to_dof_map(V)
    coords = V.tabulate_dof_coordinates()
    u_array = u.vector().get_local()
    coords = coords[v2d,:]
    u_array = u_array[v2d]
    X = coords[:,0].reshape(nx+1, ny+1)
    Y = coords[:,1].reshape(nx+1, ny+1)
    Z = u_array.reshape(nx+1, ny+1)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           rstride=1, cstride=1)
    fig.colorbar(surf)

def symdiff(l,k):
    from sympy import exp, sin, pi
    import sympy as sym
    H = lambda x: exp(-16*(x-0.5)**2)*sin(l*pi*x)
    x, y = sym.symbols('x[0], x[1]')
    u = H(x)*H(y)
    u_code = sym.printing.ccode(u)
    u_code = u_code.replace('M_PI', 'pi')
    u_D = Expression(u_code, degree=4)
    #
    K = lambda x,y: exp(sin(k*pi*x*y))
    kappa = K(x,y)
    f = sym.diff(-kappa*sym.diff(u, x), x) + \
        sym.diff(-kappa*sym.diff(u, y), y)
    f = sym.simplify(f)
    f_code = sym.printing.ccode(f)
    f_code = f_code.replace('M_PI', 'pi')
    f = Expression(f_code, degree=1)

    #print('C code for u:', u_code)
    #print('C code for f:', f_code)
    #pdb.set_trace()

    return u_D, f

def vPoisson(nx=32,ny=32,debug=1):
    # Create mesh and define function space
    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh, 'P', 1)

    #pdb.set_trace()
    #printCoords(V)

    l = 3
    k = 2
    kappa = Expression('exp(sin(k*pi*x[0]*x[1]))', degree=1, k=k)
    #kappa = Expression('exp(sin(k*pi*x[0]) * sin(k*pi*x[1]))', degree=6, k=k)

    #plt.figure(2)
    #c = plot(kappa, mesh=mesh)
    #plt.colorbar(c)

    # Define boundary condition and rhs
    u_D, f = symdiff(l,k)

    def boundary(x, on_boundary):
        return on_boundary

    bc = DirichletBC(V, u_D, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    a = kappa*dot(grad(u), grad(v))*dx
    L = f*v*dx

    A, b = assemble_system(a, L, bc)

    # Assemble linear system
    #A = assemble(a)
    #b = assemble(L)

    # Apply boundary conditions
    #bc.apply(A, b)

    if debug > 0:
        print('cond(A) =', LA.cond(A.array()))

    Acsr = csr_matrix(A.array())

    mmwrite('A.mtx',Acsr,symmetry='general')
    numpy.savetxt("A.txt", A.array())

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    # Plot solution and mesh
    #plt.figure(1)
    #c = plot(u)
    #plot(mesh)
    #plt.colorbar(c)

    u2 = project(u, V)
    surfaceplot(nx, ny, u2)

    u3 = project(u_D, V)
    surfaceplot(nx, ny, u3)

    # Compute error in L2 norm
    error_L2 = errornorm(u_D, u, 'L2')
    print('error_L2 =', errornorm(u_D, u, 'L2'))

    # Compute error in LA
    A_array = A.array()
    u_array = u.vector().get_local()
    b_array = b.get_local()
    r_array = b_array - A_array @ u_array
    print('error_max =', numpy.linalg.norm(r_array))

    # Hold plot
    plt.show()

if __name__ == '__main__':
    vPoisson(nx=31,ny=31,debug=0)
