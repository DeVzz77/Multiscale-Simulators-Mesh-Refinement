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
bound = pi
sc = 1/sqrt(bound/2)
domain = (0,bound, 0,bound)

#definitely a better way to do this 
polyDegree = [1,2,3,4]
numElements = [Int(NumDegreesofFreedom/polyDegree[1]), Int(NumDegreesofFreedom/polyDegree[2]), Int(NumDegreesofFreedom/polyDegree[3]), Int(NumDegreesofFreedom/polyDegree[4])]


model = [CartesianDiscreteModel(domain, numElements[1]), CartesianDiscreteModel(domain, numElements[2]), CartesianDiscreteModel(domain, numElements[3]), CartesianDiscreteModel(domain, numElements[4])]
cell_coords = [get_cell_coordinates(model[1]), get_cell_coordinates(model[2]), get_cell_coordinates(model[3]), get_cell_coordinates(model[4])]

Ω = [Triangulation(model[1]),Triangulation(model[2]),Triangulation(model[3]),Triangulation(model[4])]
dΩ = [Measure(Ω[1], 2*polyDegree[1] + 2), Measure(Ω[2], 2*polyDegree[2] + 2), Measure(Ω[3], 2*polyDegree[3] + 2), Measure(Ω[4], 2*polyDegree[4] + 2)]



# testspaces = Gridap.FESpaces.SingleFieldFESpace[]
# trialspaces = Gridap.FESpaces.SingleFieldFESpace[]

function GenSpacesWithPRefinement(model, polyDegree)

    #finite element definition
    reffe = ReferenceFE(lagrangian, Float64, polyDegree)
    #Test Space
    Vhhh = TestFESpace(model,reffe,dirichlet_tags="boundary")
    #Trial Space
    Uhhh= TrialFESpace(Vhhh, 0)
    return Vhhh, Uhhh
end


function getTestSpaces(model, polyDegree)
    #finite element definition
    reffe = ReferenceFE(lagrangian, Float64, polyDegree)
    #Test Space
    Vh = TestFESpace(model,reffe,dirichlet_tags="boundary")
    return Vh
end

function getTrialSpaceElement(TestSpace)
    Uh = TrialFESpace(TestSpace,0)
    return Uh
end

testspaces = [getTestSpaces(model[1], 1), getTestSpaces(model[2], 2), getTestSpaces(model[3], 3), getTestSpaces(model[4], 4)]
trialspaces = [getTrialSpaceElement(testspaces[1]), getTrialSpaceElement(testspaces[2]), getTrialSpaceElement(testspaces[3]), getTrialSpaceElement(testspaces[4])]

numSpaces = [1,2,3,4]
function GenMultipleSpaces(model, numSpaces)
    for poly in numSpaces
        #finite element definition
        reffe = ReferenceFE(lagrangian, Float64, poly)
        #Test Space
        Vhh = TestFESpace(model,reffe,dirichlet_tags="boundary")
        #Trial Space
        Uhh = TrialFESpace(Vhh, 0)
        push!(testspaces, Vhh)
        push!(trialspaces, Uhh)
    end

end




# initialise SLEPC

SlepcInitialize("-eps_target 0 -eps_nev 300 -eps_type arnoldi -eps_gen_hermitian ")
# -eps_problem_type ghep -st_type sinvert

coarseP = 1
fineP = 3

    
# VhCoarse, UhCoarse = GenSpacesWithPRefinement(model, coarseP)
# VhFine, UhFine = GenSpacesWithPRefinement(model, fineP)



# Vh = MultiFieldFESpace([VhCoarse,VhFine])
# Uh = MultiFieldFESpace([UhCoarse,UhFine])


# GenMultipleSpaces(model, numSpaces)
Vh = MultiFieldFESpace(testspaces)
Uh = MultiFieldFESpace(trialspaces)

display(Vh[2])
display(Vh)

# display(testList)
# display(trialList)





#L2 error norm
L2Norm(xh,dΩ) = ∫(xh*xh)*dΩ 

#H1 error norm
ENorm(xh,dΩ) = ∫(∇(xh)⋅∇(xh))*dΩ 

# Inner product over domain
integ(xh,yh,dΩ) = ∫(xh*yh)*dΩ





errorRefine = Vector{Vector{Float64}}()
refinementL2 = Vector{Vector{Float64}}()
refinementH1 = Vector{Vector{Float64}}()
errorCoarse = Vector{Vector{Float64}}()
errvecsL2Coarse = Vector{Vector{Float64}}()
errvecsH1Coarse = Vector{Vector{Float64}}()

diffPlot1 = plot(xscale = :log10, yscale = :log10)
diffPlot2 = plot()

#provide solutions and calculate errors for each defined space
for i in numSpaces
    #Bilinear form of stiffness matrix
    a(u,v) = ∫(∇(u)⋅∇(v))dΩ[i]
    # #Bilinear form of mass matrix
    m(u,v) = ∫(u*v)dΩ[i]


    A = assemble_matrix(a, Uh[i], Vh[i])
    M = assemble_matrix(m, Uh[i], Vh[i])
    display(A)
    display(M)



# op_a=AffineFEOperator(a,Uh[1],Vh[1]) # Generates the FE operator, holds the linear system (stiffness matrix)
# op_m=AffineFEOperator(m,Uh[1],Vh[1]) # Generates the FE operator, holds the linear system (mass matrix)
# A = op_a.op.matrix # Defines stiffness matrix
# M = op_m.op.matrix # Defines mass matrix

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
        sn=sum(integ(ev_exact,evh,dΩ[1]))
        # println("SN")
        # println(sn)
        error = ev_exact - evh*sign(sn) # Compute the error
        # display(error)
        errvecsL2[l] = sum(L2Norm(error,dΩ[1]))
        errvecsH1[l] = sum(ENorm(error,dΩ[1]))/evals[l] # Normalize by the eigenvalue


        # write to vtk
        name = "Eigenvector$(D)D$(l)_MeshRefinementP"
        writevtk(
              Ω,name,append=true,
                cellfields = [
                "evh" => evh,                           # Computed solution
                "evx" => CellField(ev_exact,Ω[1]),        # Exact solution
                "error" => CellField(error,Ω[1]),          # Exact solution
                ],
                )
        
    end


    # Plotting the errors in eigenvalues and eigenvectors
    errvals = abs.(eigenValues - evals)./evals
    push!(errorRefine,errvals)
    push!(refinementL2,errvecsL2)
    push!(refinementH1,errvecsH1)
    #display(errvals)




    if i == 1 
        push!(errorCoarse, errvals)
        display(errorCoarse)
        push!(errvecsL2Coarse, errvecsL2)
        push!(errvecsH1Coarse, errvecsH1)
                #Store errors for each refinement
        push!(errorRefine,errvals)
        push!(refinementL2,errvecsL2)
        push!(refinementH1,errvecsH1)

        eigErrorPlot = plot(errvals, xscale = :log10, yscale = :log10,label="Eigenvalue error")
        plot!(eigErrorPlot, errvecsL2; label="Eigenvector error (L2 norm)")
        plot!(eigErrorPlot, errvecsH1; label="Eigenvector error (H1 norm)")
        plot!(eigErrorPlot; legend=:bottomright)

        eigErrorPlot2 = plot(errvals, label="Eigenvalue error")
        plot!(errvecsL2, label="Eigenvector error (L2 norm)")
        plot!(errvecsH1, label="Eigenvector error (H1 norm)")
        



    else
        #Store errors for each refinement
        push!(errorRefine,errvals)
        push!(refinementL2,errvecsL2)
        push!(refinementH1,errvecsH1)

        display(errorRefine[1])

        errvalsDiff = errorRefine[1] .- errorRefine[i]
        L2Diff = refinementL2[1] .- refinementL2[i]
        H1Diff = refinementH1[1] .- errvecsH1[i]
        plot!(diffPlot1, errvalsDiff, label= "Eigenvalue Error Difference (Coarse (1) vs Refinement p degree $(i))")
        plot!(diffPlot1, L2Diff, label="Eigenvec error (L2 norm) Diff p = 1 vs $(i)")
        plot!(diffPlot1, H1Diff, label="Eigenvec error (H1 norm) Diff p = 1 vs $(i)" )
        
        plot!(diffPlot2, errvalsDiff, label= "Eigenvalue Error Difference (Coarse (1) vs Refinement p degree $(i))")
        plot!(diffPlot2, L2Diff, label="Eigenvec error (L2 norm) Diff p = 1 vs $(i)")
        plot!(diffPlot2, H1Diff, label="Eigenvec error (H1 norm) Diff p = 1 vs $(i)")

    end

    
    println("errvecsL2")
    display(errvecsL2)
    println("errvecsH1")
    display(errvecsH1)

    #Clean up, free memory
    MatDestroy(petA)
    MatDestroy(petM)
    EPSDestroy(eps)



# println("errvals")
# println(errvals)
end 


#Plot all errorvals and errorvecs on same graphs

eigErrorPlot = plot(xscale = :log10, yscale = :log10, legend=:bottomright, legendfontsize=:4)
for i in numSpaces
    plot!(eigErrorPlot, errorRefine[i],label="Eigenvalue error (p=$(i))")
    plot!(eigErrorPlot, refinementL2[i], label="Eigenvector error (L2 norm) (p=$(i))")
    plot!(eigErrorPlot, refinementH1[i], label="Eigenvector error (H1 norm) (p=$(i))")
end
png(eigErrorPlot, "ErrorsLog_P_refinement.png")


eigErrorPlot2 = plot(legend=:topright, legendfontsize=:4)
for i in numSpaces
    plot!(eigErrorPlot2, errorRefine[i], label="Eigenvalue error (p=$(i))")
    plot!(eigErrorPlot2, refinementL2[i], label="Eigenvector error (L2 norm) (p=$(i))")
    plot!(eigErrorPlot2, refinementH1[i], label="Eigenvector error (H1 norm)(p=$(i))")
end 
png(eigErrorPlot2, "Errors_P_Refinement.png")


#Plot on different graphs

for i in numSpaces
    eigErrorPlot = plot(xscale = :log10, yscale = :log10, legend=:bottomright, legendfontsize=:4)
    plot!(eigErrorPlot, errorRefine[i],label="Eigenvalue error (p=$(i))")
    plot!(eigErrorPlot, refinementL2[i], label="Eigenvector error (L2 norm) (p=$(i))")
    plot!(eigErrorPlot, refinementH1[i], label="Eigenvector error (H1 norm) (p=$(i))")
    png(eigErrorPlot, "ErrorsLog_P_refinement=$(i).png")
end




for i in numSpaces
    eigErrorPlot2 = plot(legend=:topright, legendfontsize=:4)
    plot!(eigErrorPlot2, errorRefine[i], label="Eigenvalue error (p=$(i))")
    plot!(eigErrorPlot2, refinementL2[i], label="Eigenvector error (L2 norm) (p=$(i))")
    plot!(eigErrorPlot2, refinementH1[i], label="Eigenvector error (H1 norm)(p=$(i))")
    png(eigErrorPlot2, "Errors_P_Refinement=$(i).png")
end 




display(errorRefine)
display(refinementL2)
display(refinementH1)

    #     errvalsDiff = errorRefine[1] - errvals
    #     L2Diff = refinementL2[1] - errvecsL2
    #     H1Diff = refinementH1[1] - errvecsH1
    #     plot!(diffPlot1, errvalsDiff, label= "Eigenvalue Error Difference (Coarse (1) vs Refinement p degree $(i))")
    #     plot!(diffPlot1, L2Diff, label="Eigenvec error (L2 norm) Diff p = 1 vs $(i)")
    #     plot!(diffPlot1, H1Diff, label="Eigenvec error (H1 norm) Diff p = 1 vs $(i)" )
        
    #     plot!(diffPlot2, errvalsDiff, label= "Eigenvalue Error Difference (Coarse (1) vs Refinement p degree $(i))")
    #     plot!(diffPlot2, L2Diff, label="Eigenvec error (L2 norm) Diff p = 1 vs $(i)")
    #     plot!(diffPlot2, H1Diff, label="Eigenvec error (H1 norm) Diff p = 1 vs $(i)")

# export plots to pngs
png(eigErrorPlot, "ErrorsLog.png")
png(eigErrorPlot2, "ErrorsCoarse.png")
png(diffPlot1, "diffplot1.png")
png(diffPlot2, "diffplot2.png")

SlepcFinalize()



