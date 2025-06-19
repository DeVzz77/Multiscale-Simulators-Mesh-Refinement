using Gridap 
using LinearAlgebra
using Plots
#using GridapPETSc
using PetscWrap
using SlepcWrap
using SparseArrays
#____UPDATED EIGENVALUE MESH REFINEMENT APPLICATION using PETSC/Slepc
#adapting PoissonEigen.jl

# initialise SLEPC

SlepcInitialize()


NumDegreesofFreedom = 1500
polyDegree = 2
numElements = Int(1000/polyDegree) 
bound = pi

domain = (0,bound)

model = CartesianDiscreteModel(domain,numElements)

#finite element definition
reffe = ReferenceFE(lagrangian, Float64, polyDegree)
#Test Space
Vh = FESpace(model,reffe,dirichlet_tags="boundary")
#Trial Space
Uh = TrialFESpace(Vh, 0)

Ω = Triangulation(model)

dΩ = Measure(Ω, 2*polyDegree)


#Bilinear form of stiffness matrix
a(u,v) = ∫(∇(u)⋅∇(v))dΩ
#Bilinear form of mass matrix
m(u,v) = ∫(u*v)dΩ

#Linear form
l(v) = ∫(0.0*v)dΩ

operatorA = AffineFEOperator(a,l,Uh,Vh)
operatorM = AffineFEOperator(m,l,Uh,Vh)

#stiffness matrix, A
A = operatorA.op.matrix
#mass matrix, M
M = operatorM.op.matrix
#show(stdout, "text/plain", A)
#show(stdout, "text/plain", M)
#Get matrix dimensions
rowA, columnA = size(A)
rowM, columnM = size(M)
#Get matrix values/dimensions to fill Petsc array
rowsA, columnsA,entryA = findnz(A)
rowsM, columnsM, entryM = findnz(M)

#print(rowA)
#print(columnM)
#create PETSC Matrices
petA = MatCreate()
petM = MatCreate()
MatSetSizes(petA, PETSC_DECIDE, PETSC_DECIDE, rowA,columnA)
MatSetSizes(petM, PETSC_DECIDE, PETSC_DECIDE, rowM,columnM)
MatSetFromOptions(petA)
MatSetFromOptions(petM)
MatSetUp(petA)
MatSetUp(petM)

#A_rstart, A_rend = MatGetOwnershipRange(petA)
#M_rstart, M_rend = MatGetOwnershipRange(petM)

#fill PETSC Matrices

for k in 1:length(entryA)
	MatSetValue(petA, rowsA[k]-1,columnsA[k]-1, entryA[k], INSERT_VALUES)
end

for k in 1:length(entryM)
	MatSetValue(petM, rowsM[k]-1,columnsM[k]-1, entryM[k], INSERT_VALUES)
end

#for j in 1:columnsA
#	for i in 1:rowsA
#		entry = A[i,j]
#		if entry != 0.0
#			MatSetValues(petA,i-1,j-1,entry, INSERT_VALUES)
#		end
#	end
#end

#for j in 1:columnsM
#	for i in 1:rowsM
#		entry = M[i,j]
#		if entry != 0.0
#			MatSetValues(petM,i-1,j-1,entry, INSERT_VALUES)
#		end
#	end
#end


#Assemble Matrices

MatAssemblyBegin(petA, MAT_FINAL_ASSEMBLY)
MatAssemblyBegin(petM, MAT_FINAL_ASSEMBLY)
MatAssemblyEnd(petA, MAT_FINAL_ASSEMBLY)
MatAssemblyEnd(petM, MAT_FINAL_ASSEMBLY)


#Eigenvalue solver based on SlepcWrap demo1 

eps = EPSCreate()
EPSSetOperators(eps, petA, petM)
EPSSetFromOptions(eps)
EPSSetUp(eps)

EPSSolve(eps)
EPSView(eps)

nconv = EPSGetConverged(eps)

#vals = ComplexF64[]
vals = get_eigs(eps)

for ieig in 0:nconv-1
	vpr, vpi = EPSGetEigenvalue(eps, ieig)

	val = EPSGetEigenvalue(eps,ieig)

	println("Eigenvalue: $ieig: $val")
	#@show (vpr), (vpi)
	
end

for ieig in 0:nconv - 1
	vpr, vpi, vecpr, vecpi = EPSGetEigenpair(eps,ieig,vecr, veci)
	vecs = VecGetArray(vecr)
end

#plot results:based on PoissonEigen.jl _______
Nev = length(vals)
evals = (1:Nev) .^ 2
errvals = (vals - evals)./evals
Plots.plot(errvals, xscale = :log10, yscale = :log10,title="Eigenvalue error")
plot!(legend=:topleft)
Plots.plot(errvals, title="Eigenvalue error")
ev1 = vecs[:,1]
ev2 = vecs[:,2]
ev3 = vecs[:,3]
ev4 = vecs[:,4]
ev5 = vecs[:,5]
ev6 = vecs[:,6]
Plots.plot!(ev1, label="Eigenvector 1")
Plots.plot!(ev2, label="Eigenvector 2")
Plots.plot!(ev3, label="Eigenvector 3")
Plots.plot!(ev4, label="Eigenvector 4")
Plots.plot!(ev5, label="Eigenvector 5")
Plots.plot!(ev6, label="Eigenvector 6")

#Clean up, free memory
MatDestroy(petA)
MatDestroy(petM)
EPSDestroy(eps)
SlepcFinalize()




