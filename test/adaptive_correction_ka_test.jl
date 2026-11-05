#=
    Tests for the KernelAbstractions implementation of the nearly-singular adaptive correction
    (`adaptive_correction(iop; backend)` and `adaptive_correction_plan`). Only the `CPU()`
    backend is tested here.
=#

using Test
using LinearAlgebra
using SparseArrays
using StaticArrays
using ForwardDiff
using Inti
using KernelAbstractions
using AcceleratedKernels
using ImplicitBVH
using Gmsh

const Ext = Base.get_extension(Inti, :IntiKernelAbstractionsExt)

function ellipse_quadrature(; meshsize, order, qorder)
    Inti.clear_entities!()
    msh = try
        gmsh.initialize(String[], false)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.model.occ.addDisk(0, 0, 0, 1, 0.7)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(1)
        gmsh.model.mesh.set_order(order)
        Inti.import_mesh(; dim = 2)
    finally
        gmsh.finalize()
    end
    Γ = Inti.Domain(e -> Inti.geometric_dimension(e) == 1, msh)
    return Inti.Quadrature(msh[Γ]; qorder)
end

# targets displaced from the quadrature nodes along the normal by `δ * h`
function offsurface_targets(Q, h, δs, side)
    return [(coords = q.coords + side * δ * h * q.normal, normal = q.normal) for q in Q for δ in δs]
end

relerr(A, B) = maximum(norm, nonzeros(A) .- nonzeros(B)) / maximum(norm, nonzeros(B))
same_pattern(A, B) = findnz(A)[1:2] == findnz(B)[1:2]

@testset "Adaptive integrator" begin
    # ∫₀¹ dt / ((t-a)² + d²) and ∫₀¹ log((t-a)² + d²) dt
    F1(a, d) = (atan((1 - a) / d) + atan(a / d)) / d
    G(s, d) = s * log(s^2 + d^2) - 2s + 2d * atan(s / d)
    F2(a, d) = G(1 - a, d) - G(-a, d)
    a = 0.3
    for T in (Float64, Float32), d in (1.0e-1, 1.0e-2, 1.0e-3)
        rule = Ext._embedded_rule(T)
        rtol = T == Float64 ? 1.0e-10 : 1.0e-5
        f = t -> SVector(1 / ((t[1] - T(a))^2 + T(d)^2), log((t[1] - T(a))^2 + T(d)^2))
        I, nsub, converged = Ext._adaptive_integrate(f, rule, T(rtol), zero(T), Val(256))
        @test converged
        @test eltype(I) == T
        Iex = SVector(F1(a, d), F2(a, d))
        @test norm(I - Iex) ≤ 10 * rtol * norm(Iex)
        if T == Float64
            # same algorithm as HAdaptiveIntegration: agreement up to rounding
            seg = Inti.HAdaptiveIntegration.Orthotope(0.0, 1.0)
            Iref, _ = Inti.HAdaptiveIntegration.integrate(f, seg; rtol, atol = 0.0)
            @test norm(I - Iref) ≤ 1.0e-14 * norm(Iref)
        end
    end
    # exhausting `maxsubdiv` is reported
    rule = Ext._embedded_rule(Float64)
    f = t -> 1 / ((t[1] - a)^2 + 1.0e-8)
    _, nsub, converged = Ext._adaptive_integrate(f, rule, 1.0e-12, 0.0, Val(4))
    @test nsub == 4 && !converged
    # generic numeric types: derivative with respect to the location of the near-singularity
    d = 1.0e-2
    g = a -> Ext._adaptive_integrate(t -> 1 / ((t[1] - a)^2 + d^2), rule, 1.0e-10, 0.0, Val(256))[1]
    dF1 = (a, d) -> 1 / (d^2 + a^2) - 1 / (d^2 + (1 - a)^2)
    @test ForwardDiff.derivative(g, a) ≈ dF1(a, d) rtol = 1.0e-6
    # no allocations
    function allocations(rule, a, d)
        h = t -> SVector(1 / ((t[1] - a)^2 + d^2), log((t[1] - a)^2 + d^2))
        Ext._adaptive_integrate(h, rule, 1.0e-10, 0.0, Val(256))
        return @allocated Ext._adaptive_integrate(h, rule, 1.0e-10, 0.0, Val(256))
    end
    @test allocations(rule, a, d) == 0
end

@testset "Lagrange basis" begin
    for n in (1, 3, 5)
        qrule = Inti.GaussLegendre(n)
        L = Ext._lagrange_basis(qrule, Float64)
        Lref = Inti.lagrange_basis(qrule)
        for x in (SVector(0.0), SVector(0.37), SVector(1.0))
            @test L(x) ≈ Lref(x) atol = 1.0e-12
        end
        @test all(i -> L(Inti.qcoords(qrule)[i]) ≈ (1:n .== i), 1:n)
    end
end

h = 0.1
δs = (1.0e-2, 0.1, 0.5)
Q = ellipse_quadrature(; meshsize = h, order = 3, qorder = 5)
X = vcat(offsurface_targets(Q, h, δs, -1), offsurface_targets(Q, h, δs, 1))
laplace, stokes = Inti.Laplace(; dim = 2), Inti.Stokes(; μ = 1.2, dim = 2)
kernels = (
    Inti.SingleLayerKernel(laplace),
    Inti.DoubleLayerKernel(laplace),
    Inti.AdjointDoubleLayerKernel(laplace),
    Inti.HyperSingularKernel(laplace),
    Inti.SingleLayerKernel(stokes),
    Inti.DoubleLayerKernel(stokes),
    Inti.AdjointDoubleLayerKernel(stokes),
)

@testset "Agreement with the default path (Float64)" begin
    # Same algorithm, so the results agree up to rounding errors. These are amplified by the
    # nearly-singular kernel (≈ eps * |x| / d for targets at distance d), and the KA path
    # evaluates `x - y` in element-centered coordinates, so `1e-10 ≪ rtol` is used below.
    kw = (maxdist = 3h, rtol = 1.0e-8, atol = 0.0)
    for K in kernels
        iop = Inti.IntegralOperator(K, X, Q)
        δref = Inti.adaptive_correction(iop; kw...)
        δka = Inti.adaptive_correction(iop; kw..., backend = CPU())
        @test eltype(δka) == eltype(δref)
        @test same_pattern(δka, δref)
        @test relerr(δka, δref) < 1.0e-10
    end
    # parametric (non-Lagrange) elements are used as is
    Inti.clear_entities!()
    Γ = Inti.parametric_curve(x -> SVector(cos(x), 0.7 * sin(x)), 0.0, 2π) |> Inti.Domain
    Qp = Inti.Quadrature(Γ; meshsize = h, qorder = 5)
    Xp = offsurface_targets(Qp, h, δs, 1)
    for K in (Inti.DoubleLayerKernel(laplace), Inti.SingleLayerKernel(stokes))
        iop = Inti.IntegralOperator(K, Xp, Qp)
        δref = Inti.adaptive_correction(iop; kw...)
        δka = Inti.adaptive_correction(iop; kw..., backend = CPU())
        @test same_pattern(δka, δref)
        @test relerr(δka, δref) < 1.0e-10
    end
end

@testset "Green's representation formula, $(nameof(typeof(op)))" for op in (laplace, stokes)
    # u = G(⋅, xs) * c is a solution in the interior. For x off Γ, S[∂ₙu](x) - D[u](x) is
    # equal to u(x) inside, and vanishes outside.
    xs = SVector(3.0, 3.0)
    c = op isa Inti.Stokes ? SVector(0.3, -0.7) : 1.0
    γ₀u = [Inti.SingleLayerKernel(op)(q, xs) * c for q in Q]
    γ₁u = [Inti.AdjointDoubleLayerKernel(op)(q, xs) * c for q in Q]
    for side in (-1, 1)
        Xs = offsurface_targets(Q, h, δs, side)
        u = side == -1 ? [Inti.SingleLayerKernel(op)(x, xs) * c for x in Xs] : [zero(c) for x in Xs]
        Sop = Inti.IntegralOperator(Inti.SingleLayerKernel(op), Xs, Q)
        Dop = Inti.IntegralOperator(Inti.DoubleLayerKernel(op), Xs, Q)
        S, D = Inti.assemble_matrix(Sop), Inti.assemble_matrix(Dop)
        er(δS, δD) = norm(norm.((S + δS) * γ₁u - (D + δD) * γ₀u - u), Inf) / norm(norm.(γ₀u), Inf)
        # Float64: same error as the default path, at the level of the discretization error
        kw = (maxdist = 3h, rtol = 1.0e-8, atol = 0.0)
        e_ref = er(Inti.adaptive_correction(Sop; kw...), Inti.adaptive_correction(Dop; kw...))
        e_64 = er(Inti.adaptive_correction(Sop; kw..., backend = CPU()), Inti.adaptive_correction(Dop; kw..., backend = CPU()))
        @test e_64 < 2.0e-6
        @test isapprox(e_64, e_ref; rtol = 1.0e-6)
        # Float32: error within an order of magnitude of the requested tolerance
        kw = (maxdist = 3h, rtol = 1.0e-5, atol = 0.0)
        δS = Inti.adaptive_correction(Sop; kw..., backend = CPU(), T = Float32)
        δD = Inti.adaptive_correction(Dop; kw..., backend = CPU(), T = Float32)
        @test eltype(eltype(δS)) == Float32
        @test er(δS, δD) < 10 * kw.rtol
    end
    if op isa Inti.Stokes
        # ∫_Γ G(x,y) n(y) ds_y = 0 since G is divergence-free in y
        n = [q.normal for q in Q]
        Sop = Inti.IntegralOperator(Inti.SingleLayerKernel(op), X, Q)
        S = Inti.assemble_matrix(Sop)
        @test norm(norm.(S * n), Inf) > 1.0e-3 # the correction is needed
        for (T, rtol) in ((Float64, 1.0e-8), (Float32, 1.0e-5))
            δS = Inti.adaptive_correction(Sop; maxdist = 3h, rtol, atol = 0.0, backend = CPU(), T)
            @test norm(norm.((S + δS) * n), Inf) < 1.0e-6
        end
    end
end

@testset "Argument checks" begin
    iop = Inti.IntegralOperator(Inti.SingleLayerKernel(laplace), X, Q)
    kw = (maxdist = 3h, atol = 0.0)
    # tolerance not achievable in Float32
    @test_throws ArgumentError Inti.adaptive_correction(iop; kw..., rtol = 1.0e-8, backend = CPU(), T = Float32)
    @test isnothing(Ext._check_precision(CPU(), Float32, 1.0e-5, 0.0))
    # `T` requires a backend
    @test_throws ArgumentError Inti.adaptive_correction(iop; kw..., rtol = 1.0e-5, T = Float32)
    # unsupported keyword arguments
    @test_throws ArgumentError Inti.adaptive_correction(iop; kw..., rtol = 1.0e-5, backend = CPU(), foo = 1)
    # singular interactions
    iop_sing = Inti.IntegralOperator(Inti.SingleLayerKernel(laplace), Q, Q)
    @test_throws ArgumentError Inti.adaptive_correction(iop_sing; kw..., rtol = 1.0e-5, backend = CPU())
    # tolerance not reached within `maxsubdiv`
    @test_throws ErrorException Inti.adaptive_correction(iop; kw..., rtol = 1.0e-12, backend = CPU(), maxsubdiv = 2)
    # parametric elements cannot be converted to Float32
    Inti.clear_entities!()
    Γ = Inti.parametric_curve(x -> SVector(cos(x), sin(x)), 0.0, 2π) |> Inti.Domain
    Qp = Inti.Quadrature(Γ; meshsize = h, qorder = 5)
    iop_p = Inti.IntegralOperator(Inti.SingleLayerKernel(laplace), offsurface_targets(Qp, h, δs, 1), Qp)
    @test_throws ArgumentError Inti.adaptive_correction(iop_p; kw..., rtol = 1.0e-5, backend = CPU(), T = Float32)
end

@testset "Device API" begin
    laplace_sl = Inti.SingleLayerKernel(laplace)
    plan = Inti.adaptive_correction_plan(laplace_sl, Q; backend = CPU(), maxdist = 3h, rtol = 1.0e-8, atol = 0.0)
    Xd = [(coords = x.coords, normal = x.normal) for x in X]
    # nearly-singular pairs: same as `near_interaction_list` followed by the filter on the
    # distance to the closest quadrature node
    keys, npairs = Ext._nearfield_keys(plan, Xd)
    pairs = Set(Ext._pair_from_key(k) for k in Array(keys)[1:npairs])
    msh = Inti.mesh(Q)
    E = only(Inti.element_types(msh))
    el2qtags = Inti.etype2qtags(Q, E)
    nearlist = Inti.near_interaction_list([Inti.coords(x) for x in X], msh; tol = 3h)[E]
    pairs_ref = Set{NTuple{2, Int32}}()
    for n in eachindex(nearlist), i in nearlist[n]
        dmin = minimum(j -> norm(X[i].coords - Q[j].coords), el2qtags[:, n])
        dmin ≤ 3h && push!(pairs_ref, (i, n))
    end
    @test pairs == pairs_ref
    # CSR structure and application
    δ = Inti.adaptive_correction(plan, Xd)
    δref = Inti.adaptive_correction(Inti.IntegralOperator(laplace_sl, X, Q); maxdist = 3h, rtol = 1.0e-8, atol = 0.0)
    δh = SparseMatrixCSC(δ)
    @test size(δ) == size(δref)
    @test same_pattern(δh, δref) && relerr(δh, δref) < 1.0e-10
    rowptr, colval = Array(δ.rowptr), Array(δ.colval)
    @test all(i -> issorted(colval[rowptr[i]:(rowptr[i + 1] - 1)]), 1:size(δ, 1))
    σ = rand(length(Q))
    y = rand(length(Xd))
    @test δ * σ ≈ δh * σ
    @test mul!(copy(y), δ, σ, 2.0, -0.5) ≈ 2.0 * (δh * σ) - 0.5 * y
    # targets without normals (the single layer kernel does not need them), and reuse of
    # the plan for a different set of targets
    Xs = [x.coords for x in offsurface_targets(Q, h, (0.05, 0.3), 1)]
    δs = Inti.adaptive_correction(plan, Xs)
    δs_ref = Inti.adaptive_correction(Inti.IntegralOperator(laplace_sl, Xs, Q); maxdist = 3h, rtol = 1.0e-8, atol = 0.0)
    @test relerr(SparseMatrixCSC(δs), δs_ref) < 1.0e-10
    # block (Stokes) entries
    stokes_dl = Inti.DoubleLayerKernel(stokes)
    plan = Inti.adaptive_correction_plan(stokes_dl, Q; backend = CPU(), T = Float32, maxdist = 3h, rtol = 1.0e-5, atol = 1.0e-6)
    δ = Inti.adaptive_correction(plan, Xd)
    @test eltype(δ) == SMatrix{2, 2, Float32, 4}
    σ = rand(SVector{2, Float32}, length(Q))
    @test δ * σ ≈ SparseMatrixCSC(δ) * σ
    # singular interactions are detected relative to the rounding errors of the targets: with
    # Float64 targets, a target at distance 1e-6 (≈ 8 eps(Float32) |x|) from a node is nearly
    # singular even when T = Float32, while a target on a node is singular
    plan32 = Inti.adaptive_correction_plan(laplace_sl, Q; backend = CPU(), T = Float32, maxdist = 3h, rtol = 1.0e-5, atol = 1.0e-6)
    q = Q[1]
    δn = Inti.adaptive_correction(plan32, [q.coords + 1.0e-6 * q.normal])
    @test nnz(δn) > 0
    @test_throws ArgumentError Inti.adaptive_correction(plan32, [q.coords])
    # processing the pairs sorted by cost (default on GPUs) gives the same result
    keys, npairs = Ext._nearfield_keys(plan, Xd)
    c0, v0, n0 = Ext._nearfield_values(plan, Xd, keys, npairs; sort_by_cost = false)
    c1, v1, n1 = Ext._nearfield_values(plan, Xd, keys, npairs; sort_by_cost = true)
    @test c0 == c1 && v0 == v1 && n0 == n1
    # default bound on the number of subdivisions depends on the precision
    @test Ext._val(plan.maxsubdiv) == 64
    @test Ext._val(Inti.adaptive_correction_plan(laplace_sl, Q; backend = CPU(), maxdist = 3h, rtol = 1.0e-8).maxsubdiv) == 256
    # targets far away: empty correction
    δ0 = Inti.adaptive_correction(plan, [(coords = SVector(10.0, 10.0), normal = SVector(1.0, 0.0))])
    @test nnz(δ0) == 0 && size(δ0) == (1, length(Q))
    # only meshes with a single type of line elements (here: triangles)
    Inti.clear_entities!()
    msh2 = try
        gmsh.initialize(String[], false)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.2)
        gmsh.model.occ.addDisk(0, 0, 0, 1, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        Inti.import_mesh(; dim = 2)
    finally
        gmsh.finalize()
    end
    Ω2 = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, msh2)
    Q2 = Inti.Quadrature(msh2[Ω2]; qorder = 2)
    @test_throws ArgumentError Inti.adaptive_correction_plan(laplace_sl, Q2; backend = CPU(), maxdist = 0.1, rtol = 1.0e-5)
end
