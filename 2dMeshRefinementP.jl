using Gridap 
using LinearAlgebra
using Plots
#using GridapPETSc
using PetscWrap
using SlepcWrap
using SparseArrays
using BenchmarkTools



#____UPDATED EIGENVALUE P MESH REFINEMENT APPLICATION using PETSC/Slepc in 2d
#adapting PoissonEigen.jl
# TO DO: verify that improving the space of refinement does not change coarse eigenvalue problem
# TO DO: post process eigenpairs for each P - give them as initial guesses for slepc
# TO DO: compare compute times with and without initial guesses
# TIMING could use Julia's Profiler?


# initialise SLEPC

SlepcInitialize("-eps_target 0 -eps_nev 300 -eps_type arnoldi -eps_gen_hermitian ")
# -eps_problem_type ghep -st_type sinvert
D = 1 
NumDegreesofFreedom = 5
polyDegree = 1
numElements = Int(NumDegreesofFreedom/polyDegree) 
bound = pi

sc = 1/sqrt(bound/2)

Nplot = 100

domain = (0,bound, 0, bound)

model = CartesianDiscreteModel(domain,numElements)
cell_coords = get_cell_coordinates(model)

Ω = Triangulation(model)
dΩ = Measure(Ω, 2*polyDegree + 2)

#finite element definition
reffe = ReferenceFE(lagrangian, Float64, polyDegree)

#Test Space
VhCoarse = FESpace(model,reffe,dirichlet_tags="boundary")
#Trial Space
UhCoarse = TrialFESpace(VhCoarse, 0)




#Bilinear form of stiffness matrix
a(u,v) = ∫(∇(u)⋅∇(v))dΩ
#Bilinear form of mass matrix
m(u,v) = ∫(u*v)dΩ


A = assemble_matrix(a, UhCoarse, VhCoarse)
M = assemble_matrix(m, UhCoarse, VhCoarse)
println("A")
display(A)
println("M")
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
# to time initial solve without guesses
@btime begin 

    EPSSolve(eps)
end

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
    println("EIGENVALUE")
    display(vpr)
  
    println("EIGENVEC")
    display(eigenvec)
    evh = FEFunction(VhCoarse, eigenvec) 
    # println("evh")
    # println(evh)
    ev_exact(x) = sc*sin(l*x[1])  # Exact eigenfunction
    # println("EV_EXACT(X)")
    # println(ev_exact(l))
    sn=sum(integ(ev_exact,evh,dΩ))
    # println("SN")
    # println(sn)
    error = ev_exact - evh*sign(sn) # Compute the error
    #println(error) 
     display(error)
    show(error)
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

 
# Post processing 

#refined solution
polyDegree = 2
# modelRefined = CartesianDiscreteModel(domain,numElements)
# cell_coords = get_cell_coordinates(modelRefined)



#finite element definition
reffe_refined = ReferenceFE(lagrangian, Float64, polyDegree)

#Test Space
VhRefined = FESpace(model,reffe_refined,dirichlet_tags="boundary")
#Trial Space
UhRefined = TrialFESpace(VhRefined, 0)



#Bilinear form of stiffness matrix
bref(u,v) = ∫(∇(u)⋅∇(v))dΩ
#Bilinear form of mass matrix
mref(u,v) = ∫(u*v)dΩ

Bref = assemble_matrix(bref, UhRefined, VhRefined)
Mref = assemble_matrix(mref, UhRefined, VhRefined)

# evh = FEFunction(VhCoarse, eigenvec) 
# euh = FEFunction(UhCoarse, eigenvec)

# evhref = FEFunction(VhRefined, eigenvec) 
# euhref = FEFunction(UhRefined, eigenvec)

# println("evh")
# show(evh)
# println("euh")
# show(euh)

# println("evhref")
# show(evhref)
# println("euhref")
# show(euhref)

# op_b=AffineFEOperator(bref,UhRefined,VhRefined) # Generates the FE operator, holds the linear system (stiffness matrix)
# op_m=AffineFEOperator(mref,UhRefined,VhRefined) # Generates the FE operator, holds the linear system (mass matrix)
# Bref = op_a.op.matrix # Defines stiffness matrix
# Mref = op_m.op.matrix # Defines mass matrix


bpost(u,v) = ∫(∇(u)⋅∇(v))dΩ
mpost(u,v) = ∫(u*v)dΩ

Bpost = assemble_matrix(bpost, UhCoarse, VhRefined ) 
Mpost = assemble_matrix(mpost, UhCoarse, VhRefined )

# op_b_post=AffineFEOperator(bpost,UhRefined,VhCoarse) # Generates the FE operator, holds the linear system (stiffness matrix)
# op_m_post=AffineFEOperator(mpost,UhRefined,VhCoarse) # Generates the FE operator, holds the linear system (mass matrix)
# Bpost = op_a.op.matrix # Defines stiffness matrix
# Mpost = op_m.op.matrix # Defines mass matrix


#error after refinement

errorRefined = Vector{Vector{Float64}}()

println("typeof(Bref) = ", typeof(Bref), ", size(Bref) = ", size(Bref))
println("typeof(Mpost) = ", typeof(Mpost), ", size(Mpost) = ", size(Mpost))
println("typeof(Bpost) = ", typeof(Bpost), ", size(Bpost) = ", size(Bpost))



for ieig in 0:nconv-1
    i = ieig +1
    vpr, vpi, vecpr, vecpi = EPSGetEigenpair(eps,ieig,vecr, veci)
    eigenvec = vec2array(vecpr)
    
    println("typeof(eigenvec) = ", typeof(eigenvec), ", size(eigenvec) = ", size(eigenvec))
    ei = (Bref' * (((vpr*Mpost) - Bpost))) * eigenvec
    show(ei)
    
    push!(errorRefined, ei)
end


#extra
#to approximate eigenfunction better

Vi = Vector{Vector{Float64}}()
for ieig in 0: nconv - 1
    i = ieig + 1
    vpr, vpi, vecpr, vecpi = EPSGetEigenpair(eps,ieig,vecr, veci)
    eigenvec = vec2array(vecpr)

    vi = vpr * Bpost' * Mpost * eigenvec
    print("V")
    println(i)
    show(vi)
    push!(Vi, vi)
end

println("typeof(VI) = ", typeof(Vi), ", size(VI) = ", size(Vi))

#to approximate eigenvalues better
Mi = Vector{Float64}()
for ieig in 0: nconv - 1
    i = ieig + 1
    vpr, vpi, vecpr, vecpi = EPSGetEigenpair(eps,ieig,vecr, veci)
     
    #dimension mismatch error for B and M
    mi = dot(Vi[i], Bref * Vi[i]) / dot(Vi[i], Mref * Vi[i])
    println("MI")
    show(mi)
    push!(Mi, mi)

end 


#display(errorRefined) 

#Clean up, free memory
MatDestroy(petA)
MatDestroy(petM)
EPSDestroy(eps)
SlepcFinalize()