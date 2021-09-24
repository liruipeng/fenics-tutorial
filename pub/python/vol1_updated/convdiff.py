from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy
from numpy import linalg as LA
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
from sympy import exp, sin, cos, pi, sqrt
import sympy as sym
import pdb

def sympy2expression(u_sym,degree=1,printit=0):
    u_code = sym.printing.ccode(u_sym)
    u_code = u_code.replace('M_PI', 'pi')
    if printit:
        print('C code:', u_code)
    u_expr = Expression(u_code, degree=degree)
    return u_expr

def u0_boundary(x, on_boundary):
    return on_boundary

# When $h < \varepsilon$, the solution is stable.
def solve_transport_2dfem(n, epsilon,kx=1,ky=1,ax=1,ay=1,alpha=pi/4):
    # Construct the mesh
    mesh = UnitSquareMesh(n, n)

    x, y = sym.symbols('x[0], x[1]')
    kappa_lam = lambda x,y: 1.1 + cos(kx*pi*(cos(alpha)*(x-0.5)-sin(alpha)*(y-0.5)+0.5+ax)) * cos(ky*pi*(sin(alpha)*(x-0.5)+cos(alpha)*(y-0.5)+0.5+ax))
    kappa_sym = kappa_lam(x,y)
    kappa = sympy2expression(kappa_sym, degree=1)

    # Define trial and test functions
    V = FunctionSpace(mesh, 'P', 1)
    u = TrialFunction(V)
    v = TestFunction(V)

    f = Constant(1.)
    b = Constant((1., 1.))

    # Define boundary conditions
    u0 = Constant(0)
    bc = DirichletBC(V, u0, u0_boundary)

    # Define variational form
    a = epsilon * kappa * dot(grad(u), grad(v)) * dx + dot(b, grad(u)) * v * dx
    L = f * v * dx

    u = Function(V)
    solve(a == L, u, bc)

    return u

# The Galerkin least squares (GLS) method combines Galerkin method and the least squares finite elements to stablize the solution. It can be considered as a systematic way for introducing artificial diffusion or regularization terms.
def solve_transport_2dfem_gls(n, epsilon, C=1, p=1, kx=1,ky=1,ax=1,ay=1,alpha=pi/4):
    # Construct the mesh
    mesh = UnitSquareMesh(n, n)

    # Define trial and test functions
    V = FunctionSpace(mesh, 'P', p)

    x, y = sym.symbols('x[0], x[1]')
    kappa_lam = lambda x,y: 1.1 + cos(kx*pi*(cos(alpha)*(x-0.5)-sin(alpha)*(y-0.5)+0.5+ax)) * cos(ky*pi*(sin(alpha)*(x-0.5)+cos(alpha)*(y-0.5)+0.5+ax))
    kappa_sym = kappa_lam(x,y)
    kappa = sympy2expression(kappa_sym, degree=1)

    plot(kappa, mesh=mesh,title='kappa')
    plt.show()

    u = TrialFunction(V)
    v = TestFunction(V)

    f = Constant(1.)
    b = Constant((1., 1.))

    # Define boundary conditions
    u0 = Constant(0)
    bc = DirichletBC(V, u0, u0_boundary)

    # Define variational form
    a = epsilon * kappa * dot(grad(u), grad(v)) * dx + dot(b, grad(u)) * v * dx
    L = f * v * dx

    # Note that div(grad(u))=0 for linear finite elements
    Lu = lambda u : -epsilon * div(grad(u)) + dot(b, grad(u))

    h = 2*Circumradius(mesh)
    sigma = h * Constant(C)
    a_stablized = a + sigma * Lu(u) * Lu(v) * dx
    L_stablized = L + sigma * f * Lu(v) * dx

    A, b = assemble_system(a_stablized, L_stablized, bc)

    u = Function(V)
    solve(a_stablized == L_stablized, u, bc)

    return u

u = solve_transport_2dfem(n=31, epsilon=1/16, kx=8, ky=8, ax=0.1)
plot(u)
plt.show()

