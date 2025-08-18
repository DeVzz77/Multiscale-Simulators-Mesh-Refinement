using Gridap 
using LinearAlgebra
using Plots
#using GridapPETSc
using PetscWrap
using SlepcWrap
using SparseArrays




#____UPDATED EIGENVALUE MESH REFINEMENT APPLICATION using PETSC/Slepc
#adapting PoissonEigen.jl
D = 1

NumDegreesofFreedom = 300
polyDegree = 1
numElements = Int(NumDegreesofFreedom/polyDegree)
bound = pi
sc = 1/sqrt(bound/2)
domain = (0,bound)

model = CartesianDiscreteModel(domain, numElements)
cell_coords = get_cell_coordinates(model)

Ω = Triangulation(model)
dΩ = Measure(Ω, 2*polyDegree + 2)

function GenSpacesWithPRefinement(model, polyDegree)

    #finite element definition
    reffe = ReferenceFE(lagrangian, Float64, polyDegree)
    #Test Space
    Vh = TestFESpace(model,reffe,dirichlet_tags="boundary")
    #Trial Space
    Uh = TrialFESpace(Vh, 0)
    return Vh, Uh
end

# initialise SLEPC

SlepcInitialize("-eps_target 0 -eps_nev 300 -eps_type arnoldi -eps_gen_hermitian ")
# -eps_problem_type ghep -st_type sinvert

coarseP = 1
fineP = 3

VhCoarse, UhCoarse = GenSpacesWithPRefinement(model, coarseP)
VhFine, UhFine = GenSpacesWithPRefinement(model, fineP)

Vh = MultiFieldFESpace([VhCoarse,VhFine])
Uh = MultiFieldFESpace([UhCoarse,UhFine])




#Bilinear form of stiffness matrix
a(u,v) = ∫(∇(u)⋅∇(v))dΩ
#Bilinear form of mass matrix
m(u,v) = ∫(u*v)dΩ
#Linear form
l(v) = ∫(0.0*v)dΩ

op_a=AffineFEOperator(a,Uh,Vh) # Generates the FE operator, holds the linear system (stiffness matrix)
op_m=AffineFEOperator(m,Uh,Vh) # Generates the FE operator, holds the linear system (mass matrix)
A = op_a.op.matrix # Defines stiffness matrix
M = op_m.op.matrix # Defines mass matrix
# A = assemble_matrix(a, Uh[1], Vh[1])
# M = assemble_matrix(m, Uh[1], Vh[1])
display(A)
display(M)



#Get matrix dimensions
rowA, columnA = size(A)
rowM, columnM = size(M)
#Get matrix values/dimensions to fill Petsc array
rowsA, columnsA, entryA = findnz(A)
rowsM, columnsM, entryM = findnz(M)

#create PETSC Matrices
petA = MatCreate()
petM = MatCreate()
MatSetSizes(petA, PETSC_DECIDE, PETSC_DECIDE, rowA,columnA)
MatSetSizes(petM, PETSC_DECIDE, PETSC_DECIDE, rowM,columnM)
MatSetFromOptions(petA)
MatSetFromOptions(petM)
MatSetUp(petA)
MatSetUp(petM)


#fill PETSC Matrices

for k in 1:length(entryA)
	MatSetValue(petA, rowsA[k]-1,columnsA[k]-1, entryA[k], INSERT_VALUES)
end

for k in 1:length(entryM)
	MatSetValue(petM, rowsM[k]-1,columnsM[k]-1, entryM[k], INSERT_VALUES)
end


#Assemble Matrices

MatAssemblyBegin(petA, MAT_FINAL_ASSEMBLY)
MatAssemblyBegin(petM, MAT_FINAL_ASSEMBLY)
MatAssemblyEnd(petA, MAT_FINAL_ASSEMBLY)
MatAssemblyEnd(petM, MAT_FINAL_ASSEMBLY)

# display(petA)
# display(petM)

#Eigenvalue solver based on SlepcWrap demo1 

eps = EPSCreate()
EPSSetOperators(eps, petA, petM)
EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE)
EPSSetFromOptions(eps)
EPSSetUp(eps)

EPSSolve(eps)
EPSView(eps)

nconv = EPSGetConverged(eps)
#nconv = 100
#vals = ComplexF64[]
vals = get_eigs(eps)
eigenValues = zeros(nconv)
#show(vals)

eigenpairs = get_eigenpair(eps,1)
#show(eigenpairs)

vecr, veci = MatCreateVecs(petA)

#THIS WORKS
for ieig in 0:nconv - 1
 	vpr, vpi, vecpr, vecpi = EPSGetEigenpair(eps,ieig,vecr, veci)
    eigenValues[ieig+1] = vpr
end



#plot results: 
Nev = nconv

# println("NEV:")
# println(Nev)



evals = (1:Nev) .^ 2
# println(evals)

#L2 error norm
L2Norm(xh,dΩ) = ∫(xh*xh)*dΩ 

#H1 error norm
ENorm(xh,dΩ) = ∫(∇(xh)⋅∇(xh))*dΩ 

# Inner product over domain
integ(xh,yh,dΩ) = ∫(xh*yh)*dΩ


# Relative error in eigenvalues
errvecsL2 = zeros(Nev) # Vector to store eigenvector errors
errvecsH1 = zeros(Nev) # Vector to store eigenvector errors


# println("EIGEN VALUES")
# println(eigenValues)

# println("Nconv")
# println(nconv)
for ieig in 0:nconv-1
    l = ieig + 1
    vpr, vpi, vecpr, vecpi = EPSGetEigenpair(eps,ieig,vecr, veci)
    eigenvec = vec2array(vecpr)
    # println("L VALUE")
    # println(l)
  
    #println("EIGENVEC")
    #display(eigenvec)
    evh = FEFunction(Vh[1], eigenvec) 
    # println("evh")
    # println(evh)
    ev_exact(x) = sc*sin(l*x[1])  # Exact eigenfunction
    # println("EV_EXACT(X)")
    # println(ev_exact(l))
    sn=sum(integ(ev_exact,evh,dΩ))
    # println("SN")
    # println(sn)
    error = ev_exact - evh*sign(sn) # Compute the error
    # display(error)
    errvecsL2[l] = sum(L2Norm(error,dΩ))
    errvecsH1[l] = sum(ENorm(error,dΩ))/evals[l] # Normalize by the eigenvalue

    
end

println("errvecsL2")
display(errvecsL2)
println("errvecsH1")
display(errvecsH1)


# Plotting the errors in eigenvalues and eigenvectors
errvals = abs.(eigenValues - evals)./evals
# println("errvals")
# println(errvals)

eigErrorPlot = plot(errvals, xscale = :log10, yscale = :log10,label="Eigenvalue error")
plot!(eigErrorPlot, errvecsL2; label="Eigenvector error (L2 norm)")
plot!(eigErrorPlot, errvecsH1; label="Eigenvector error (H1 norm)")
plot!(eigErrorPlot; legend=:bottomright)
png(eigErrorPlot, "plot1.png")
eigErrorPlot2 = plot(errvals, label="Eigenvalue error")
plot!(errvecsL2, label="Eigenvector error (L2 norm)")
plot!(errvecsH1, label="Eigenvector error (H1 norm)")
png(eigErrorPlot2, "plot2.png")

 
#Clean up, free memory
MatDestroy(petA)
MatDestroy(petM)
EPSDestroy(eps)
SlepcFinalize()


