#=
    Benchmark of the nearly-singular adaptive correction: Inti's default (threaded) path
    versus the device pipeline (`adaptive_correction_plan` + `adaptive_correction(plan, X)`)
    on the KernelAbstractions CPU backend (POCL) and on a GPU.

    Run from the repository root with e.g.

        julia --project=benchmark -t auto benchmark/adaptive_correction_ka.jl

    The GPU is selected with `INTI_BENCH_BACKEND=metal` (default) or `INTI_BENCH_BACKEND=cuda`
    (requires `CUDA` in the benchmark environment: `] add CUDA`). Mesh sizes can be set with
    e.g. `INTI_BENCH_MESHSIZES=1e-2,2.5e-3`.

    The geometry is an ellipse meshed by Gmsh with curved (P3) Lagrange elements, and the
    targets are off-surface points at distances `d ∈ [1e-2 h, h]` (log-uniform) from the
    boundary, two per boundary quadrature node (one on each side).

    Variants are labeled by backend, precision `T` of the integration, and precision of the
    targets `X` (Float64 whenever the backend supports it).

    Timed stages (seconds; minimum over repetitions):
    - plan:   one-time setup for the (fixed) source geometry, including transfer to the device
    - h2d:    transfer of the targets to the device (only if they live on the host)
    - search: nearly-singular pairs (target BVH, traversal, filter, sort)
    - csr:    row pointers of the CSR matrix
    - kernel: adaptive integration, filling the CSR values
    - check:  convergence check
    - step:   search + csr + kernel + check, i.e. one correction with targets on the device
              (for Inti: the whole correction, on the host)
    - d2h:    transfer of the result to the host as a `SparseMatrixCSC`
    - mul:    one application of the correction on the device (`mul!`); for Inti, the
              (single-threaded) host `SparseMatrixCSC` product

    Accuracy is the largest error in an entry of the correction, relative to the largest
    entry (`scale`), measured against a Float64 reference computed with `rtol = 1e-10`. With
    targets in Float32, pairs at distance ≈ `maxdist` may be included or not differently from
    the reference; such entries count as errors.
    All variants use the same tolerances: `rtol` and `atol = 0.1 * rtol * scale`.
=#

using Inti
using KernelAbstractions
using AcceleratedKernels
using ImplicitBVH
using Gmsh
using StaticArrays
using SparseArrays
using LinearAlgebra
using Printf
using Random

const BACKEND_NAME = get(ENV, "INTI_BENCH_BACKEND", "metal")
if BACKEND_NAME == "metal"
    @eval using Metal
    gpu_backend() = MetalBackend()
elseif BACKEND_NAME == "cuda"
    @eval using CUDA
    gpu_backend() = CUDABackend()
else
    error("unknown INTI_BENCH_BACKEND = $BACKEND_NAME")
end

const Ext = Base.get_extension(Inti, :IntiKernelAbstractionsExt)

function ellipse_quadrature(; meshsize, order = 3, qorder = 5)
    Inti.clear_entities!()
    msh = try
        gmsh.initialize(String[], false)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
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

function near_targets(Q, h; rng = MersenneTwister(0))
    X = NamedTuple{(:coords, :normal), Tuple{SVector{2, Float64}, SVector{2, Float64}}}[]
    for q in Q, side in (-1, 1)
        d = h * 10^(-2 * rand(rng))
        push!(X, (coords = q.coords + side * d * q.normal, normal = q.normal))
    end
    return X
end

# minimum wall time over `n` runs, after one warm-up run; `f` must synchronize
function besttime(f, n)
    f()
    return minimum(_ -> @elapsed(f()), 1:n)
end

function device_stages(K, Q, X, maxdist, rtol, atol, backend, T, nrep)
    sync(x) = (KernelAbstractions.synchronize(backend); x)
    mkplan = () -> Inti.adaptive_correction_plan(K, Q; backend, T, maxdist, rtol, atol)
    tplan = besttime(() -> sync(mkplan()), 1)
    plan = mkplan()
    # targets in the precision of the plan's centers (Float64 if supported by the backend)
    S = eltype(eltype(plan.centers))
    Xh = [(coords = SVector{2, S}(x.coords), normal = SVector{2, S}(x.normal)) for x in X]
    Xd = Ext._to_device(backend, Xh)
    m = length(X)
    keys, npairs = Ext._nearfield_keys(plan, Xd)
    _, _, nsub = Ext._nearfield_values(plan, Xd, keys, npairs)
    δ = Inti.adaptive_correction(plan, Xd)
    σ = Ext._to_device(backend, rand(eltype(δ) <: SMatrix ? SVector{2, T} : T, length(Q)))
    y = KernelAbstractions.allocate(backend, Base.promote_op(*, eltype(δ), eltype(σ)), m)
    t = (
        plan = tplan,
        h2d = besttime(() -> sync(Ext._to_device(backend, Xh)), nrep),
        search = besttime(() -> sync(Ext._nearfield_keys(plan, Xd)), nrep),
        csr = besttime(() -> sync(Ext._csr_rowptr(plan, keys, npairs, m)), nrep),
        kernel = besttime(() -> sync(Ext._nearfield_values(plan, Xd, keys, npairs)), nrep),
        check = besttime(() -> Ext._check_convergence(plan, nsub), nrep),
        step = besttime(() -> sync(Inti.adaptive_correction(plan, Xd)), nrep),
        d2h = besttime(() -> SparseMatrixCSC(δ), nrep),
        mul = besttime(() -> sync(mul!(y, δ, σ)), max(nrep, 10)),
    )
    nsub = Array(nsub)
    return t, SparseMatrixCSC(δ), sum(nsub) / length(nsub)
end

# largest entry of `δ - δref` relative to the largest entry of `δref`; entries present in only
# one of the matrices (e.g. pairs at distance ≈ `maxdist`, decided differently because of
# rounding) count as errors
function entry_error(δ, δref)
    return maximum(norm, nonzeros(δ - δref); init = 0.0) / maximum(norm, nonzeros(δref))
end

const COLUMNS = (:plan, :h2d, :search, :csr, :kernel, :check, :step, :d2h, :mul)

function header()
    @printf("%-11s %-19s %-6s %8s |", "kernel", "variant", "rtol", "pairs")
    foreach(c -> @printf(" %8s", c), COLUMNS)
    @printf(" | %8s %5s\n", "error", "nsub")
    return println("-"^(50 + 9 * length(COLUMNS) + 17))
end

function row(kname, variant, rtol, npairs, t, err, nsub)
    @printf("%-11s %-19s %-6s %8d |", kname, variant, @sprintf("%.0e", rtol), npairs)
    foreach(c -> @printf(" %8s", haskey(t, c) ? @sprintf("%.2e", t[c]) : "—"), COLUMNS)
    return @printf(" | %8.1e %5.1f\n", err, nsub)
end

function run_benchmark(; meshsizes, nrep = 3, maxdist_factor = 3)
    gpu = gpu_backend()
    println("Julia threads: ", Threads.nthreads(), "   GPU backend: ", gpu)
    laplace, stokes = Inti.Laplace(; dim = 2), Inti.Stokes(; μ = 1.0, dim = 2)
    kernels = (
        "Laplace SL" => Inti.SingleLayerKernel(laplace),
        "Laplace DL" => Inti.DoubleLayerKernel(laplace),
        "Stokes SL" => Inti.SingleLayerKernel(stokes),
        "Stokes DL" => Inti.DoubleLayerKernel(stokes),
    )
    gpu_name = BACKEND_NAME == "cuda" ? "CUDA" : "Metal"
    for h in meshsizes
        Q = ellipse_quadrature(; meshsize = h)
        X = near_targets(Q, h)
        maxdist = maxdist_factor * h
        println()
        println("h = $h: $(length(Q) ÷ 3) elements, $(length(Q)) sources, $(length(X)) targets, maxdist = $maxdist")
        header()
        for (kname, K) in kernels
            iop = Inti.IntegralOperator(K, X, Q)
            δref = Inti.adaptive_correction(iop; maxdist, rtol = 1.0e-10, atol = 0.0, backend = CPU(), maxsubdiv = 1024)
            npairs = nnz(δref) ÷ 3
            scale = maximum(norm, nonzeros(δref))
            for rtol in (1.0e-5, 1.0e-8)
                atol = 0.1 * rtol * scale
                # Inti's default path (host)
                δ = Inti.adaptive_correction(iop; maxdist, rtol, atol)
                σ = rand(eltype(δ) <: SMatrix ? SVector{2, Float64} : Float64, length(Q))
                t = (
                    step = besttime(() -> Inti.adaptive_correction(iop; maxdist, rtol, atol), nrep),
                    mul = besttime(() -> δ * σ, max(nrep, 10)),
                )
                row(kname, "Inti (threads)", rtol, npairs, t, entry_error(δ, δref), NaN)
                variants = Any[("POCL", CPU(), Float64)]
                KernelAbstractions.supports_float64(gpu) && push!(variants, (gpu_name, gpu, Float64))
                if rtol ≥ 1.0e-5
                    push!(variants, ("POCL", CPU(), Float32))
                    push!(variants, (gpu_name, gpu, Float32))
                end
                for (bname, backend, T) in variants
                    # targets are given in Float64 when the backend supports it
                    X_T = KernelAbstractions.supports_float64(backend) ? 64 : 32
                    vname = "$bname T=F$(8sizeof(T)) X=F$X_T"
                    try
                        t, δ, nsub = device_stages(K, Q, X, maxdist, rtol, atol, backend, T, nrep)
                        row(kname, vname, rtol, npairs, t, entry_error(δ, δref), nsub)
                    catch err
                        msg = (s = sprint(showerror, err); r = findfirst("Reason", s); isnothing(r) ? first(split(s, "\n")) : s[first(r):min(end, first(r) + 300)])
                        @printf("%-11s %-19s %-6s   FAILED: %s\n", kname, vname, @sprintf("%.0e", rtol), msg)
                    end
                end
            end
        end
    end
    return
end

meshsizes = haskey(ENV, "INTI_BENCH_MESHSIZES") ?
    parse.(Float64, split(ENV["INTI_BENCH_MESHSIZES"], ",")) : [1.0e-2, 2.5e-3, 6.25e-4, 1.5625e-4]
run_benchmark(; meshsizes)
