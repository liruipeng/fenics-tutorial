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
from sympy import exp, sin, cos, pi, sqrt
import sympy as sym
import pdb

def printCoords(V, mesh, ordering='Mesh', printfile=False):
    # note that the ordering of dof can be different from mesh vertices given by
    # coords = mesh.coordinates()
    # the mapping between can be got by
    # v2d = vertex_to_dof_map(V)
    if ordering == 'Mesh':
        coords = mesh.coordinates()
    elif ordering == 'dof':
        coords = V.tabulate_dof_coordinates()

    if printfile:
        ndofs  = V.dim()
        #nnodes = coords.shape[0]
        dim = coords.shape[1]
        print("Printing coordinates for mesh with",ndofs,"nodes and dimension",dim)
        header  = "%24d %24d" %(ndofs, dim)
        if(dim == 2):
            numpy.savetxt("coords.txt", coords, header=header, comments='', fmt="%24.15e %24.15e")
        elif(dim == 3):
            numpy.savetxt("coords.txt", coords, header=header, comments='', fmt="%24.15e %24.15e %24.15e")

    return coords


def is_pos_def(A):
    if numpy.array_equal(A, A.T):
        try:
            numpy.linalg.cholesky(A)
            return True
        except numpy.linalg.LinAlgError:
            return False
    else:
        return False

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

def sympydiff(kappa, u):
    x, y = sym.symbols('x[0], x[1]')
    f = sym.diff(-kappa*sym.diff(u, x), x) + \
        sym.diff(-kappa*sym.diff(u, y), y)
    f = sym.simplify(f)
    return f

def sympy2expression(u_sym,degree=1,printit=0):
    u_code = sym.printing.ccode(u_sym)
    u_code = u_code.replace('M_PI', 'pi')
    if printit:
        print('C code:', u_code)
    u_expr = Expression(u_code, degree=degree)
    return u_expr

def vPoisson(nx=32,ny=32,knownf=0,knownu=1,kx=1,ky=1,ax=1,ay=1,alpha=pi/4,debug=1,seeplot=0):
    # Create mesh and define function space
    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh, 'P', 1)

    #coords = printCoords(V, mesh, ordering='Mesh', printfile=True)

    x, y = sym.symbols('x[0], x[1]')

    if knownu:
        # K: rotated ellipse
        kappa_lam = lambda x,y: 0.01 + (((x-0.5)*cos(alpha)+(y-0.5)*sin(alpha))**2)/(ax**2) + (((x-0.5)*sin(alpha)-(y-0.5)*cos(alpha))**2)/(ay**2)
        kappa_sym = kappa_lam(x,y)
        # Assume exact solution is sin(kx*pi*x)*sin(ky*pi*y)
        # Define boundary condition (exact solution) and rhs
        u_lam = lambda x,y: sin(kx*pi*x)*sin(ky*pi*y)
        u_sym = u_lam(x,y)
        f_sym = sympydiff(kappa_sym, u_sym)
        u_D = sympy2expression(u_sym, degree=4)
        f = sympy2expression(f_sym, degree=1, printit=0)
    elif knownf:
        # K:
        kappa_lam = lambda x,y: (1.01 + sin(kx*pi*x+pi/2) * sin(ky*pi*y+pi/2))*exp(-(ax*x+ay*y))
        kappa_sym = kappa_lam(x,y)
        # Assume f is 1000*[(x-0.5)^2+(y-0.5)^2] and bdc is constant 1.0
        #f = Expression('1000*(pow(x[0]-0.5,2)+pow(x[1]-0.5,2))',degree=1)
        f = Expression('16*exp(-16*(pow(x[0]-0.5,2)+pow(x[1]-0.5,2)))',degree=1)
        u_D = Constant(1.0)

    kappa = sympy2expression(kappa_sym, degree=1)

    if seeplot > 1:
        surfaceplot(nx, ny, project(kappa, V))
        plt.figure()
        c = plot(kappa, mesh=mesh,title='kappa')
        plt.colorbar(c)
        if knownu:
            plt.figure()
            c = plot(u_D, mesh=mesh,title='u_D')
            plt.colorbar(c)
        plt.figure()
        c = plot(f, mesh=mesh,title='f')
        plt.colorbar(c)
        surfaceplot(nx, ny, project(f, V))
        plt.show()

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
        print('A is SPD ?', is_pos_def(A.array()))
        print('cond_2(A) = %.2e' % LA.cond(A.array()))

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    if seeplot > 1:
        # Plot solution and exact solution
        plt.figure()
        c = plot(u)
        #plot(mesh)
        plt.colorbar(c)
        surfaceplot(nx, ny, project(u, V))
        if knownu:
            surfaceplot(nx, ny, project(u_D, V))
        plt.show()

    if debug > 0:
        if knownu:
            # Compute error in L2 norm
            error_L2 = errornorm(u_D, u, 'L2')
            print('error_L2 =', errornorm(u_D, u, 'L2'))
        # Compute error in LA
        A_array = A.array()
        u_array = u.vector().get_local()
        b_array = b.get_local()
        r_array = b_array - A_array @ u_array
        print('error_max =', numpy.linalg.norm(r_array))

    if seeplot == 1:
        plt.figure()
        c = plot(kappa, mesh=mesh,title='kappa')
        plt.colorbar(c)
        plt.figure()
        c = plot(u,title='u')
        plt.colorbar(c)
        if knownf == 0:
            plt.figure()
            c = plot(f, mesh=mesh,title='f')
            plt.colorbar(c)
        #plt.show()

    # output CSR
    #Acsr = csr_matrix(A.array())
    #mmwrite('A.mtx',Acsr,symmetry='general')
    # output dense
    #numpy.savetxt("A.txt", A.array())

    kappa_vertex = kappa.compute_vertex_values(mesh)
    u_vertex = u.compute_vertex_values()
    f_vertex = f.compute_vertex_values(mesh)

    return kappa_vertex, u_vertex, f_vertex

if __name__ == '__main__':

    # prob 0: known f (fixed), changing K, solve u
    # prob 1: manufactured u (variable), compute f, changing K, solve u
    prob = 0

    nx=31
    ny=31
    n = (nx+1)*(ny+1)

    # output file names
    Kfn = "Kappa.txt"
    ffn = "F.txt"
    ufn = "U.txt"
    save_begin = 0
    save_freq = 10000

    na = 5
    nh = 3
    nk = 32

    kappa_all = numpy.empty((0,n))
    f_all = numpy.empty((0,n))
    u_all = numpy.empty((0,n))

    if prob == 1:
        #  solution
        alphas = [*range(0, na)]
        alphas = [x/na*pi for x in alphas]
        heights = [*range(1, nh+1)]
        heights=2**numpy.array(heights)
        #
        for alpha in alphas:
            for ay in heights:
                for kx in range(1,nk+1):
                    for ky in range(1,nk+1):
                        kappa, u, f = vPoisson(nx=nx,ny=ny,knownf=0,knownu=1,kx=kx,ky=ky,ax=1,ay=ay,alpha=alpha,debug=0,seeplot=0)
                        kappa_all = numpy.vstack((kappa_all, kappa.reshape(1,-1)))
                        u_all = numpy.vstack((u_all, u.reshape(1,-1)))
                        f_all = numpy.vstack((f_all, f.reshape(1,-1)))
    else:
        omega = numpy.linspace(1, 16, num=nk)
        decay = numpy.linspace(0, 1, num=na)
        counter = 0

        #debug
        #for ax in decay:
            #for ay in decay:
                #for kx in omega:
                    #for ky in omega:
                       #if (counter == 14368):
                          #print('Prob %6d: kx %f ky %f ax %f ay %f' % (counter, kx, ky, ax, ay))
                          #kappa, u, f = vPoisson(nx=31,ny=31,knownf=1,knownu=0,kx=kx,ky=ky,ax=ax,ay=ay,debug=1,seeplot=1)
                          #pdb.set_trace()
                       #counter = counter + 1
        #pdb.set_trace()
        #debug end

        for ax in decay:
            for ay in decay:
                for kx in omega:
                    for ky in omega:
                        counter = counter + 1
                        if counter <= save_begin:
                           continue
                        print('Prob %6d: kx %f ky %f ax %f ay %f' % (counter, kx, ky, ax, ay))
                        kappa, u, f = vPoisson(nx=nx,ny=ny,knownf=1,knownu=0,kx=kx,ky=ky,ax=ax,ay=ay,debug=0,seeplot=0)
                        kappa_all = numpy.vstack((kappa_all, kappa.reshape(1,-1)))
                        u_all = numpy.vstack((u_all, u.reshape(1,-1)))
                        f_all = numpy.vstack((f_all, f.reshape(1,-1)))
                        if counter % save_freq == 0:
                           with open(Kfn, "a") as f:
                              numpy.savetxt(f, kappa_all)
                           with open(ffn, "a") as f:
                              numpy.savetxt(f, f_all)
                           with open(ufn, "a") as f:
                              numpy.savetxt(f, u_all)
                           kappa_all = numpy.empty((0,n))
                           u_all = numpy.empty((0,n))
                           f_all = numpy.empty((0,n))

                        #if counter == 7:
                        #    input("Press any key to continue the program")
                        #    plt.show()

    with open(Kfn, "a") as f:
       numpy.savetxt(f, kappa_all)
    with open(ffn, "a") as f:
       numpy.savetxt(f, f_all)
    with open(ufn, "a") as f:
       numpy.savetxt(f, u_all)

    #plt.show()
    #pdb.set_trace()

##############################################
# Some interesting functions (on [0,1]x[0,1])
##############################################

#Marr (the Mexican Hat)
#kappa = Expression('1/(pi*pow(k,4))*(1-(pow(x[0]-0.5,2)+pow(x[1]-0.5,2))/(2*pow(k,2)))*exp(-(pow(x[0]-0.5,2)+pow(x[1]-0.5,2))/(2*pow(k,2)))', degree=1, k=k)

# 2-D Gaussian
#sx = 0.25
#sy = 0.25
#ro = 0.0
#kappa_lam = lambda x,y: 1/(2*pi*sx*sy*sqrt(1-(ro**2)))*exp(-1/(2*(1-(ro**2)))*(((x-0.5)/sx)**2-2*ro*(x-0.5)/sx*(y-0.5)/sy+((y-0.5)/sy)**2))

# rotated ellipse
#a = 0.25
#b = 0.5
#alpha = pi/4
#kappa_lam = lambda x,y: (((x-0.5)*cos(alpha)+(y-0.5)*sin(alpha))**2)/(a**2) + (((x-0.5)*sin(alpha)-(y-0.5)*cos(alpha))**2)/(b**2)
