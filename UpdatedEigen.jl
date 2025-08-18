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

SlepcInitialize("-eps_target 0 -eps_nev 20")

D = 1
NumDegreesofFreedom = 300
polyDegree = 3
numElements = Int(NumDegreesofFreedom/polyDegree) 
bound = pi

sc = 1/sqrt(bound/2)

Nplot = 4

domain = (0,bound)

model = CartesianDiscreteModel(domain,numElements)
cell_coords = get_cell_coordinates(model)

Ω = Triangulation(model)
dΩ = Measure(Ω, 2*polyDegree + 2)

#finite element definition
reffe = ReferenceFE(lagrangian, Float64, polyDegree)

#Test Space
Vh = FESpace(model,reffe,dirichlet_tags="boundary")
#Trial Space
Uh = TrialFESpace(Vh, 0)




#Bilinear form of stiffness matrix
a(u,v) = ∫(∇(u)⋅∇(v))dΩ
#Bilinear form of mass matrix
m(u,v) = ∫(u*v)dΩ

#Linear form
#l(v) = ∫(0.0*v)dΩ

#operatorA = AffineFEOperator(a,l,Uh,Vh)
#operatorM = AffineFEOperator(m,l,Uh,Vh)
A = assemble_matrix(a, Uh, Vh)
M = assemble_matrix(m, Uh, Vh)
#print(A)
#print(M)

#stiffness matrix, A
#A = operatorA.op.matrix
#mass matrix, M
#M = operatorM.op.matrix
dense = true
test = true
#show(stdout, "text/plain", A)
#show(stdout, "text/plain", M)
#Get matrix dimensions
rowA, columnA = size(A)
rowM, columnM = size(M)
#Get matrix values/dimensions to fill Petsc array
rowsA, columnsA, entryA = findnz(A)
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
EPSSetWhichEigenpairs(eps, EPS_SMALLEST_MAGNITUDE)
EPSSetFromOptions(eps)
EPSSetUp(eps)

EPSSolve(eps)
EPSView(eps)

nconv = EPSGetConverged(eps)
#nconv = 100
#vals = ComplexF64[]
vals = get_eigs(eps)
global eigenValues = zeros(nconv)
#show(vals)

eigenpairs = get_eigenpair(eps,1)
#show(eigenpairs)

#eigenvectors = get_eigenvector(eps,1)
#show(eigenvectors)

# for ieig in 0:nconv-1
# 	eig = get_eig(eps, ieig)
# 	show(eig)
# end

#for ieig in 0:nconv-1
	#vpr, vpi = EPSGetEigenvalue(eps, ieig)

	#eigenValuess = EPSGetEigenvalue(eps,ieig)

	#println("Eigenvalue: $ieig: $eigenValues")
	#@show √(vpr), √(vpi)
#	show(vpr)
#	show(vpi)
	
#end
vecr, veci = MatCreateVecs(petA)

#THIS WORKS
for ieig in 0:nconv - 1
	vpr, vpi, vecpr, vecpi = EPSGetEigenpair(eps,ieig,vecr, veci)

    eigenValues[ieig+1] = vpr
    vecC = vec2array(vecpr)
    show(vpr)
    show(vecC)
	#@show (vecpr), (vecpi)
	#vecs = VecGetArray(vecr)
end



#plot results: 
Nev = Int(floor(nconv/2)*2)
print("NEV:::")
show(Nev)
#errvals = (vals - evals)./evals

evals = (1:Nev) .^ 2
show(evals)
#L2 error norm
L2Norm(xh,dΩ) = ∫(xh*xh)*dΩ 
show(L2Norm)
#H1 error norm
ENorm(xh,dΩ) = ∫(∇(xh)⋅∇(xh))*dΩ 
show(ENorm)
# Inner product over domain
integ(xh,yh,dΩ) = ∫(xh*yh)*dΩ
show(integ)

# Relative error in eigenvalues
errvecsL2 = zeros(Nev) # Vector to store eigenvector errors
errvecsH1 = zeros(Nev) # Vector to store eigenvector errors

realvecs, imagvecs = SlepcWrap.eigenvectors2mat(eps, 1)
 #show(realvecs)
# vecRange = get_range(realvecs)
# vecRangeMin, vecRangeMax = get_range(realvecs)
# show(max)
# show(vecRange)
# juliaVecs = PetscWrap.vec2array(realvecs)

# #create vector to store + convert to array
# tempVector = create(Vec,comm)
# setSizes(tempVector, PETSC_DECIDE, vecRange)
show(nconv)
for ieig in 0:nconv-1
    l = ieig + 1
    vpr, vpi, vecpr, vecpi = EPSGetEigenpair(eps,ieig,vecr, veci)
    eigenvec = vec2array(vecpr)
    print("L VALUE")
    show(l)
    #eigenvec, imagvec = SlepcWrap.eigenvectors2mat(eps,l)
    print("EIGENVEC")
    show(eigenvec)
    #vecRange = get_urange(eigenvec)
    #show(vecRange)
    #vecRangeMin, vecRangeMax = get_range(eigenvec)
    #tempEigenvector, tmpImag = MatCreateVecs(eigenvec)
    #print("TEMP EIGEN VECTORS::::::::")
    #show(tempEigenvector)
    #eigenvectors = PetscWrap.vec2array(tempEigenvector)
    #eigenvectors =VecGetArray(tempEigenvector)
    #show(eigenvectors)
    #for i in vecRange
     #   eigenvectors =  
    #end 
    evh = FEFunction(Vh, eigenvec) 
    show(evh)
    ev_exact(x) = sc*sin(l*x[1])  # Exact eigenfunction
    show(ev_exact)
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
                "evx" => CellField(ev_exact,Ω),        # Exact solution
                "error" => CellField(error,Ω),          # Exact solution
                ],
                )
    end

end


#eigs = get_eigenvalues(eps)
#show(eigs)
#eigenVV = vec2array(eigs)
#show (eigenVV)
show("EIGEN VALUES")
show(eigenValues)
# Plotting the errors in eigenvalues and eigenvectors
errvals = abs.(eigenValues - evals)./evals

#error value alone
plot3 = plot(1:length(errvals)-1, errvals; yscale=:log10, marker=:o)
png(plot3, "plot3.png")
show("ERROR VALUES")
show(errvals)
eigErrorPlot = plot(errvals, xscale = :log10, yscale = :log10,label="Eigenvalue error")
show("ErrorVecsL2")
show(errvecsL2)
plot!(eigErrorPlot, errvecsL2; label="Eigenvector error (L2 norm)")
show("ErrorVecsH1")
show(errvecsH1)
plot!(eigErrorPlot, errvecsH1; label="Eigenvector error (H1 norm)")
plot!(eigErrorPlot; legend=:bottomright)
display(eigErrorPlot)
png(eigErrorPlot, "plot1.png")
eigErrorPlot2 = plot(errvals, label="Eigenvalue error")
plot!(errvecsL2, label="Eigenvector error (L2 norm)")
plot!(errvecsH1, label="Eigenvector error (H1 norm)")
png(eigErrorPlot2, "plot2.png")
display(eigErrorPlot2)

eigErrorPlot4 = plot(errvecsL2, label="Eigenvector error (L2 norm)")
plot!(errvecsH1, label="Eigenvector error (H1 norm)")
png(eigErrorPlot4, "plot4.png")
display(eigErrorPlot4)
#plot results:based on PoissonEigen.jl _______
# Nev = length(vals)
# evals = (1:Nev) .^ 2
# errvals = (vals - evals)./evals
# Plots.plot(errvals, xscale = :log10, yscale = :log10,title="Eigenvalue error")
# plot!(legend=:topleft)
# Plots.plot(errvals, title="Eigenvalue error")
# ev1 = vecs[:,1]
# ev2 = vecs[:,2]
# ev3 = vecs[:,3]
# ev4 = vecs[:,4]
# ev5 = vecs[:,5]
# ev6 = vecs[:,6]
# Plots.plot!(ev1, label="Eigenvector 1")
# Plots.plot!(ev2, label="Eigenvector 2")
# Plots.plot!(ev3, label="Eigenvector 3")
# Plots.plot!(ev4, label="Eigenvector 4")
# Plots.plot!(ev5, label="Eigenvector 5")
# Plots.plot!(ev6, label="Eigenvector 6")

if false
    ev1 = get_eigenvector(eps,1)
    ev2 = get_eigenvector(eps,2)
    ev3 = get_eigenvector(eps,3)
    ev4 = get_eigenvector(eps,4)
    ev5 = get_eigenvector(eps,5)
    ev6 = get_eigenvector(eps,6)
    display(Plots.plot(ev1, label="Eigenvector 1"))
    display(Plots.plot!(ev2, label="Eigenvector 2"))
    display(Plots.plot!(ev3, label="Eigenvector 3"))
    display(Plots.plot!(ev4, label="Eigenvector 4"))
    display(Plots.plot!(ev5, label="Eigenvector 5"))
    display(Plots.plot!(ev6, label="Eigenvector 6"))
 end

 
#Clean up, free memory
MatDestroy(petA)
MatDestroy(petM)
EPSDestroy(eps)
SlepcFinalize()




gui()