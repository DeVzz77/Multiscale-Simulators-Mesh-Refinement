
using Gridap
  using Gridap.Algebra, Gridap.CellData, Gridap.Geometry, Gridap.Adaptivity, Gridap.ReferenceFEs

using SparseArrays
using LinearAlgebra, Arpack
using Plots
using Printf

using DataStructures

function difference_in_eigenpairs(Nev, Λ, eV, Λ_dense, Vh, dΩ)

  L2Norm(xh,dΩ) = ∫(xh*xh)*dΩ # L2 norm of the error
  ENorm(xh,dΩ) = ∫(∇(xh)⋅∇(xh))*dΩ # H1 norm of the error
  integ(xh,yh,dΩ) = ∫(xh*yh)*dΩ # Inner product over the domain

  # Relative error in eigenvalues
  diffvecsL2 = zeros(Nev) # Vector to store eigenvector difference
  diffvecsH1 = zeros(Nev) # Vector to store eigenvector difference
  diffvals   = zeros(Nev) # Vector to store eigenvalue difference
  for l in 1:Nev
    
    ievh_dense  = FEFunction(Vh, Λ_dense.vectors[:,l])
    ievh_sparse = FEFunction(Vh, eV[:,l])

    sn=sum(integ(ievh_dense,ievh_sparse,dΩ))

    error = ievh_dense - ievh_sparse *sign(sn) # Compute difference between dense and sparse eigenvectors
    diffvecsL2[l] = sum(L2Norm(error,dΩ))
    diffvecsH1[l] = sum(ENorm(error,dΩ))/Λ # Normalize by the eigenvalue

    diffvals[l] = abs(Λ_dense.values[l] - Λ[l])/Λ_dense.values[l]

    writevtk(
        Ω,"eigvecDiff$(l-1)",append=false,
        cellfields = [
          "dense" => ievh_dense,               # Computed eigenvector (dense)
          "sparse" => ievh_sparse,             # Computed eigenvector (sparse)
          "difference" => CellField(error,Ω),       # Difference
        ],
      )

  end

  return diffvals, diffvecsL2, diffvecsH1
end


# Coarse solution

NDoF = 300 # DoF
k = 3 # Polynomial degree
Nel = Int(NDoF/k) # Number of elements

L = pi
sc = 1/sqrt(L/2)

Nplot =100 # Number of plots to save

model=CartesianDiscreteModel((0,L),(Nel))
cell_coords = get_cell_coordinates(model) # Get the coordinates of the cells
Ω=Triangulation(model) # Creat Triangulation of the model
dΩ=Measure(Ω,2*k+2) # Measure (quadrature) over Ω, 2 indicates the order that is integrated exactly

reffe=ReferenceFE(lagrangian,Float64,k) # Finite element definition
Vh=FESpace(model,reffe,dirichlet_tags="boundary")  # Test space + BCs
Uh=TrialFESpace(Vh,0) # Trial space + analytic condition to impose on the boundary

a(u,v)=∫(∇(u)⋅∇(v))dΩ # Bilinear form corresponding to the stiffness matrix
m(u,v)=∫(u*v)dΩ # Bilinear form corresponding to the mass matrix

# op_a=AffineFEOperator(a,l,Uh,Vh) # Generates the FE operator, holds the linear system (stiffness matrix)
# op_m=AffineFEOperator(m,l,Uh,Vh) # Generates the FE operator, holds the linear system (mass matrix)
# A = op_a.op.matrix # Defines stiffness matrix
# M = op_m.op.matrix # Defines mass matrix

A = assemble_matrix(a, Uh, Vh)
M = assemble_matrix(m, Uh, Vh)

## length(A[:,1]) # Number of rows in the matrix
Nev = Int(floor(length(A[:,1])/2)*2) # Number of eigenvalues to compute
dense = true # Use dense matrix for eigenvalue computation
test  = true # Test the eigenvalue computation with eigs
if dense
    Adense=Array(A) # Convert to dense matrix
    Mdense=Array(M) # Convert to dense matrix
    Λ_dense = eigen(Adense, Mdense)
    if test
        #eigenvalues, eV = eigenvectors     
        Λ, eV = eigs(A, M, nev = Nev, which = :SM)  # SM = smallest magnitude 
        diffvals, diffvecsL2, diffvecsH1 = difference_in_eigenpairs(Nev, Λ, eV, Λ_dense, Vh, dΩ)
        # Plotting difference in eigenvalues and eigenvectors from dense and sparse computation
        plot(title="Relative error in eigenpairs btn dense & sparse solvers", xaxis="Eigenvalue index", yaxis="Error", legend=:bottomright)
        plot!(diffvals, xscale = :log10, yscale = :log10,label="Eigenvalue error")
        plot!(diffvecsL2, label="Eigenvector error (L2 norm)")
        plot!(diffvecsH1, label="Eigenvector error (H1 norm)")
        plot(title="Relative error in eigenpairs btn dense & sparse solvers", xaxis="Eigenvalue index", yaxis="Error", legend=:topleft)
        plot(diffvals, label="Eigenvalue error")
        plot!(diffvecsL2, label="Eigenvector error (L2 norm)")
        plot!(diffvecsH1, label="Eigenvector error (H1 norm)")
    end
    Λ = Λ_dense.values[1:Nev] # Extract the first Nev eigenvalues
    eV = Λ_dense.vectors[:,1:Nev] # Extract the first Nev eigenvectors
else
    Λ, eV = eigs(A, M, nev = Nev, which = :SM)  # SM = smallest magnitude 
end

evals = (1:Nev) .^ 2
L2Norm(xh,dΩ) = ∫(xh*xh)*dΩ # L2 norm of the error 
ENorm(xh,dΩ) = ∫(∇(xh)⋅∇(xh))*dΩ # H1 norm of the error
integ(xh,yh,dΩ) = ∫(xh*yh)*dΩ # Inner product over the domain
# Relative error in eigenvalues
errvecsL2 = zeros(Nev) # Vector to store eigenvector errors
errvecsH1 = zeros(Nev) # Vector to store eigenvector errors
for l in 1:Nev
    evh = FEFunction(Vh, eV[:,l]) 
    ev_exact(x) = sc*sin(l*x[1])  # Exact eigenfunction
    sn=sum(integ(ev_exact,evh,dΩ))
    error = ev_exact - evh*sign(sn) # Compute the error
    errvecsL2[l] = sum(L2Norm(error,dΩ))
    errvecsH1[l] = sum(ENorm(error,dΩ))/evals[l] # Normalize by the eigenvalue
    if l < Nplot
        name = "Eigenvector$(D)D$(l)"
        writevtk(
              Ω,name,append=false,
                cellfields = [
                "evh" => evh,                           # Computed solution
                "evx" => CellField(ev_exactf,Ω),        # Exact solution
                "error" => CellField(error,Ω),          # Exact solution
                ],
                )
    end

end


errvals = abs.(Λ - evals)./evals
plot(errvals, xscale = :log10, yscale = :log10,label="Eigenvalue error")
plot!(errvecsL2, label="Eigenvector error (L2 norm)")
plot!(errvecsH1, label="Eigenvector error (H1 norm)")
plot!(legend=:bottomright)
plot(errvals, label="Eigenvalue error")
plot!(errvecsL2, label="Eigenvector error (L2 norm)")
plot!(errvecsH1, label="Eigenvector error (H1 norm)")

if false
    ev1 = eV[:,1]
    ev2 = eV[:,2]
    ev3 = eV[:,3]
    ev4 = eV[:,4]
    ev5 = eV[:,5]
    ev6 = eV[:,6]
    Plots.plot(ev1, label="Eigenvector 1")
    Plots.plot!(ev2, label="Eigenvector 2")
    Plots.plot!(ev3, label="Eigenvector 3")
    Plots.plot!(ev4, label="Eigenvector 4")
    Plots.plot!(ev5, label="Eigenvector 5")
    Plots.plot!(ev6, label="Eigenvector 6")
 end

 


