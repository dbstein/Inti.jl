#=
    Tests for the KernelAbstractions implementation of the nearly-singular adaptive correction
    (`adaptive_correction(iop; backend)`). Only the `CPU()` backend is tested here.
=#

using Test
using LinearAlgebra
using SparseArrays
using StaticArrays
using ForwardDiff
using Inti
using KernelAbstractions
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
