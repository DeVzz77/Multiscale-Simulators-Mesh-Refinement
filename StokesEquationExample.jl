using Gridap 

#Stokes equation tutorial reference 

n = 100
domain = (0,1,0,1)
partition = (n,n)
model = CartesianDiscreteModel(domain, partition)

labels = get_face_labeling(model)
add_tag_from_tags!(labels,"diri1",[6,])
add_tag_from_tags!(labels,"diri0",[1,2,3,4,5,7,8])

order = 2
reffeᵤ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffeₚ = ReferenceFE(lagrangian,Float64,order-1;space=:P)

V = TestFESpace(model,reffeᵤ,labels=labels,dirichlet_tags=["diri0","diri1"],conformity=:H1)
Q = TestFESpace(model,reffeₚ,conformity=:L2,constraint=:zeromean)
Y = MultiFieldFESpace([V,Q])

println(V)
println(Q)
println(Y)

u0 = VectorValue(0,0)
u1 = VectorValue(1,0)
U = TrialFESpace(V,[u0,u1])
P = TrialFESpace(Q)
X = MultiFieldFESpace([U,P])

println(u0)
println(u1)
println(U)
println(P)
println(X)

degree = order
Ωₕ = Triangulation(model)
dΩ = Measure(Ωₕ,degree)

f = VectorValue(0.0,0.0)
a((u,p),(v,q)) = ∫( ∇(v)⊙∇(u) - (∇⋅v)*p + q*(∇⋅u) )dΩ
l((v,q)) = ∫( v⋅f )dΩ

op = AffineFEOperator(a,l,X,Y)

uh, ph = solve(op)

writevtk(Ωₕ,"results stokes",order=2,cellfields=["uh_stokes"=>uh,"ph_stokes"=>ph])using Gridap 
