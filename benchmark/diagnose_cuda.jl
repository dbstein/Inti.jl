#=
    Diagnostics for the KernelAbstractions corrections on CUDA. Run from the repository root:

        julia --project=benchmark -t 8 benchmark/diagnose_cuda.jl [stage]

    with `stage` one of `kernels`, `pocl32`, `cuda64`, `cuda32` (default: `kernels`). Each stage
    prints a line before starting, so that a crash can be located.
=#

using Inti, KernelAbstractions, AcceleratedKernels, ImplicitBVH, CUDA, Gmsh, StaticArrays, SparseArrays, LinearAlgebra, Printf
const KA = KernelAbstractions
const Ext = Base.get_extension(Inti, :IntiKernelAbstractionsExt)

say(args...) = (println(args...); flush(stdout))

stage = isempty(ARGS) ? "kernels" : ARGS[1]
say("stage: ", stage, "   threads: ", Threads.nthreads())
CUDA.versioninfo()

# 1. kernels evaluated on the device vs on the host, in Float64
@kernel function eval_kernel!(out, K, @Const(xs), @Const(ys))
    i = @index(Global, Linear)
    @inbounds out[i] = K(xs[i], ys[i])
end

function check_kernels()
    for dim in (2, 3)
        n = 10_000
        xs = [(coords = randn(SVector{dim, Float64}), normal = normalize(randn(SVector{dim, Float64}))) for _ in 1:n]
        ys = [(coords = x.coords + 1.0e-3 * randn(SVector{dim, Float64}), normal = normalize(randn(SVector{dim, Float64}))) for x in xs]
        for op in (Inti.Laplace(; dim), Inti.Stokes(; μ = 1.0, dim)), K in (Inti.SingleLayerKernel(op), Inti.DoubleLayerKernel(op))
            ref = [K(xs[i], ys[i]) for i in 1:n]
            out = CuArray{eltype(ref)}(undef, n)
            eval_kernel!(CUDABackend())(out, K, CuArray(xs), CuArray(ys); ndrange = n)
            err = maximum(i -> norm(Array(out)[i] - ref[i]) / norm(ref[i]), 1:n)
            @printf("  %dD %-24s max relative difference device vs host (Float64): %.1e\n", dim, string(nameof(typeof(K)), " ", nameof(typeof(op))), err)
        end
    end
    return
end

function sphere_quadrature(; meshsize, order = 2, qorder = 4)
    Inti.clear_entities!()
    msh = try
        gmsh.initialize(String[], false)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        gmsh.model.occ.addSphere(0, 0, 0, 1)
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.set_order(order)
        Inti.import_mesh(; dim = 3)
    finally
        gmsh.finalize()
    end
    Γ = Inti.Domain(e -> Inti.geometric_dimension(e) == 2, msh)
    return Inti.Quadrature(msh[Γ]; qorder)
end

# 2. one 3D correction with a given backend and precision, compared with the default path
function check_correction(backend, T)
    h = 0.2
    Q = sphere_quadrature(; meshsize = h)
    X = [(coords = q.coords + s * δ * h * q.normal, normal = q.normal) for q in Q for δ in (1.0e-2, 0.1, 0.5) for s in (-1, 1)]
    K = Inti.SingleLayerKernel(Inti.Laplace(; dim = 3))
    kw = T == Float32 ? (maxdist = 2h, rtol = 1.0e-5, atol = 1.0e-6) : (maxdist = 2h, rtol = 1.0e-6, atol = 0.0)
    say("  default path…")
    δref = Inti.adaptive_correction(Inti.IntegralOperator(K, X, Q); kw...)
    say("  plan…")
    plan = Inti.adaptive_correction_plan(K, Q; backend, T, kw...)
    S = eltype(eltype(plan.centers))
    Xd = Ext._to_device(backend, [(coords = SVector{3, S}(x.coords), normal = SVector{3, S}(x.normal)) for x in X])
    say("  search…")
    keys, npairs = Ext._nearfield_keys(plan, Xd)
    say("  integration ($npairs pairs, tiers $(Ext._tiers(plan, SVector{6, T})))…")
    _, _, nsub = Ext._nearfield_values(plan, Xd, keys, npairs)
    KA.synchronize(backend)
    say("  full correction…")
    δ = SparseMatrixCSC(Inti.adaptive_correction(plan, Xd))
    err = maximum(norm, nonzeros(δ - δref); init = 0.0) / maximum(norm, nonzeros(δref))
    ns = Array(nsub)
    @printf("  max entry error vs default path: %.1e   (nsub mean %.1f, max %d)\n", err, sum(abs, ns) / length(ns), maximum(abs, ns))
    return
end

if stage == "kernels"
    check_kernels()
elseif stage == "pocl32"
    check_correction(CPU(), Float32)
elseif stage == "cuda64"
    check_correction(CUDABackend(), Float64)
elseif stage == "cuda32"
    check_correction(CUDABackend(), Float32)
else
    error("unknown stage $stage")
end
say("done")
