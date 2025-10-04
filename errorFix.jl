using Gridap
using LinearAlgebra, Arpack
using Plots


#Coarse solution 

NDoF = 4
k = 2 # Polynomial degree
Nel = Int(NDoF/k) # number of elements

L = pi
sc = 1/sqrt(L/2)

Nplot = 100

model=CartesianDiscreteModel((0,L),(Nel))
cell_coords = get_cell_coordinates(model) # Get the coordinates of the cells
Ω=Triangulation(model) # Creat Triangulation of the model
dΩ=Measure(Ω,2*k+2) # Measure (quadrature) over Ω, 2 indicates the order that is integrated exactly

reffe=ReferenceFE(lagrangian,Float64,k) # Finite element definition
Vh=FESpace(model,reffe,dirichlet_tags="boundary")  # Test space + BCs
Uh=TrialFESpace(Vh,0) # Trial space + analytic condition to impose on the boundary


a(u,v)=∫(∇(u)⋅∇(v))dΩ # Bilinear form corresponding to the stiffness matrix
m(u,v)=∫(u*v)dΩ # Bilinear form corresponding to the mass matrix



A = assemble_matrix(a, Uh, Vh)
M = assemble_matrix(m, Uh, Vh)
println("A")
display(A)
println("M")
display(M)

# show("ALTERNATE")
# l(v) = ∫(0.0*v)dΩ

# op_a=AffineFEOperator(a,l,Uh,Vh) # Generates the FE operator, holds the linear system (stiffness matrix)
# op_m=AffineFEOperator(m,l,Uh,Vh) # Generates the FE operator, holds the linear system (mass matrix)
# AA = op_a.op.matrix # Defines stiffness matrix
# MM = op_m.op.matrix # Defines mass matrix

# show(AA)
# show(MM)

Nev = Int(floor(length(A[:,1])/2)*2) # Number of eigenvalues to compute
Λ, eV = eigs(A, M, nev = Nev, which = :SM)  # SM = smallest magnitude 
display("Eigenvalues")
display(Λ)
display("Eigenvectors")
display(eV)
# println("num eigenvalues")
# println(Nev)
evals = (1:Nev) .^ 2
# println("error values")
# println(evals)
L2Norm(xh,dΩ) = ∫(xh*xh)*dΩ # L2 norm of the error 
ENorm(xh,dΩ) = ∫(∇(xh)⋅∇(xh))*dΩ # H1 norm of the error
integ(xh,yh,dΩ) = ∫(xh*yh)*dΩ # Inner product over the domain
# Relative error in eigenvalues
errvecsL2 = zeros(Nev) # Vector to store eigenvector errors
errvecsH1 = zeros(Nev) # Vector to store eigenvector errors
for l in 1:Nev
    # println("eigenvectors for l")
    # display(eV[:,l])
    evh = FEFunction(Vh, eV[:,l]) 
    ev_exact(x) = sc*sin(l*x[1])  # Exact eigenfunction
    #println("ev_exact")
    #display(ev_exact(l))
    sn=sum(integ(ev_exact,evh,dΩ))
    # println("sn")
    # println(sn)
    # println("sign sn")
    # println(sign(sn))
    # println("evh*sign(sn)")
    # println(evh*sign(sn))
    error = ev_exact - evh*sign(sn) 
    # println("error")
    # display(error)
    # Compute the error
    errvecsL2[l] = sum(L2Norm(error,dΩ))
    errvecsH1[l] = sum(ENorm(error,dΩ))/evals[l] # Normalize by the eigenvalue
end 

println("errvecsL2")
display(errvecsL2)
println("errvecsH1")
display(errvecsH1)

errvals = abs.(Λ - evals)./evals
# println("errvals")
# println(errvals)
