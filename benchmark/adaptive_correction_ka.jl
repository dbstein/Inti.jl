#=
    Benchmark of the nearly-singular adaptive correction: Inti's default (threaded) path
    versus the KernelAbstractions implementation on the CPU (POCL) and on Metal.

    Run from the repository root with e.g.

        julia --project=benchmark -t auto benchmark/adaptive_correction_ka.jl

    The geometry is an ellipse meshed by Gmsh with curved (P3) Lagrange elements, and the
    targets are off-surface points at distances `d ∈ [1e-2 h, h]` (log-uniform) from the
    boundary, two per boundary quadrature node (one on each side).

    For the KernelAbstractions variants the following stages are timed separately:
    - host:     near-interaction list, pair filtering, conversion to `T` (host)
    - h2d:      transfer of the kernel inputs to the device
    - kernel:   the correction kernel (inputs and outputs on the device)
    - d2h:      transfer of the output to the host
    - assemble: convergence check and sparse matrix assembly (host)
    `resident` is the kernel time alone (data already on the device), `end-to-end` is the sum
    of all stages (data on the host).

    Accuracy is the largest error in an entry of the correction, relative to the largest
    entry (`scale`), measured against a Float64 reference computed with `rtol = 1e-10`.

    All variants use the same tolerances: `rtol` and `atol = 0.1 * rtol * scale`. A purely
    relative tolerance is not attainable in Float32 for interactions whose integral is small
    compared to the integrand (e.g. double-layer kernels for targets tangential to an element).
=#

using Inti
using KernelAbstractions
using Metal
using Gmsh
using StaticArrays
using SparseArrays
using LinearAlgebra
using Printf
using Random

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

# minimum wall time over `n` runs, after one warm-up run
function besttime(f, n)
    f()
    return minimum(_ -> @elapsed(f()), 1:n)
end

function ka_stages(iop, maxdist, rtol, atol, backend, T, maxsubdiv, nrep)
    host_prep = () -> Ext._ka_host_prepare(iop, maxdist, rtol, atol, backend, T, maxsubdiv, true)
    hp = host_prep()
    prep = Ext._ka_to_device(hp)
    out = Ext._ka_allocate_output(prep)
    Ext._ka_nearfield!(out, prep)
    oh = Ext._ka_to_host(out)
    t = (
        host = besttime(host_prep, nrep),
        h2d = besttime(() -> Ext._ka_to_device(hp), nrep),
        kernel = besttime(() -> Ext._ka_nearfield!(out, prep), nrep),
        d2h = besttime(() -> Ext._ka_to_host(out), nrep),
        assemble = besttime(() -> Ext._ka_assemble(prep, oh), nrep),
    )
    nsub = reduce(vcat, [o.nsub for o in oh])
    return t, Ext._ka_assemble(prep, oh), nsub
end

function entry_error(δ, δref)
    findnz(δ)[1:2] == findnz(δref)[1:2] || return NaN
    return maximum(norm, nonzeros(δ) .- nonzeros(δref)) / maximum(norm, nonzeros(δref))
end

function header()
    @printf(
        "%-22s %-14s %-6s %8s | %9s %9s %9s %9s %9s | %10s %10s | %9s %6s\n",
        "kernel", "variant", "rtol", "pairs", "host", "h2d", "kernel", "d2h", "assemble",
        "resident", "end-to-end", "error", "nsub"
    )
    return println("-"^150)
end

function row(kname, variant, rtol, npairs, t, err, nsub)
    st(x) = isnothing(x) ? "—" : @sprintf("%.2e", x)
    return @printf(
        "%-22s %-14s %-6s %8d | %9s %9s %9s %9s %9s | %10s %10s | %9.1e %6.1f\n",
        kname, variant, @sprintf("%.0e", rtol), npairs,
        st(t.host), st(t.h2d), st(t.kernel), st(t.d2h), st(t.assemble),
        st(t.resident), st(t.total), err, nsub
    )
end

function run_benchmark(; meshsizes, nrep = 3, maxdist_factor = 3)
    println("Julia threads: ", Threads.nthreads(), "   Metal device: ", Metal.device())
    laplace, stokes = Inti.Laplace(; dim = 2), Inti.Stokes(; μ = 1.0, dim = 2)
    kernels = (
        "Laplace SL" => Inti.SingleLayerKernel(laplace),
        "Laplace DL" => Inti.DoubleLayerKernel(laplace),
        "Stokes SL" => Inti.SingleLayerKernel(stokes),
        "Stokes DL" => Inti.DoubleLayerKernel(stokes),
    )
    for h in meshsizes
        Q = ellipse_quadrature(; meshsize = h)
        X = near_targets(Q, h)
        nel = length(Q) ÷ 3
        maxdist = maxdist_factor * h
        println()
        println("h = $h: $nel elements, $(length(Q)) sources, $(length(X)) targets, maxdist = $maxdist")
        header()
        for (kname, K) in kernels
            iop = Inti.IntegralOperator(K, X, Q)
            # Float64 reference
            δref = Inti.adaptive_correction(iop; maxdist, rtol = 1.0e-10, atol = 0.0, backend = CPU(), maxsubdiv = 1024)
            npairs = nnz(δref) ÷ 3
            scale = maximum(norm, nonzeros(δref))
            for rtol in (1.0e-5, 1.0e-8)
                atol = 0.1 * rtol * scale
                # Inti's default path
                δ = Inti.adaptive_correction(iop; maxdist, rtol, atol)
                tt = besttime(() -> Inti.adaptive_correction(iop; maxdist, rtol, atol), nrep)
                t = (host = nothing, h2d = nothing, kernel = nothing, d2h = nothing, assemble = nothing, resident = nothing, total = tt)
                row(kname, "Inti (threads)", rtol, npairs, t, entry_error(δ, δref), NaN)
                variants = Any[("KA POCL F64", CPU(), Float64, 256)]
                if rtol ≥ 1.0e-5
                    push!(variants, ("KA POCL F32", CPU(), Float32, 256))
                    push!(variants, ("KA Metal F32", MetalBackend(), Float32, 256))
                    push!(variants, ("KA Metal F32/64", MetalBackend(), Float32, 64))
                end
                for (vname, backend, T, maxsubdiv) in variants
                    t, δ, nsub = ka_stages(iop, maxdist, rtol, atol, backend, T, maxsubdiv, nrep)
                    t = merge(t, (resident = t.kernel, total = sum(t)))
                    row(kname, vname, rtol, npairs, t, entry_error(δ, δref), sum(nsub) / length(nsub))
                end
            end
        end
    end
    return
end

meshsizes = haskey(ENV, "INTI_BENCH_MESHSIZES") ?
    parse.(Float64, split(ENV["INTI_BENCH_MESHSIZES"], ",")) : [1.0e-2, 2.5e-3, 6.25e-4]
run_benchmark(; meshsizes)
