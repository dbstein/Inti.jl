#=
    Diagnostics for the KernelAbstractions corrections on a GPU (CUDA or Metal). Run from the
    repository root:

        julia --project=benchmark -t 8 benchmark/diagnose_gpu.jl [stage]

    The GPU is selected with `INTI_DIAG_BACKEND=cuda` or `INTI_DIAG_BACKEND=metal` (default:
    Metal on macOS, CUDA otherwise). Stages:
    - `kernels`: kernels evaluated on the GPU vs on the host (Float64 if supported, else Float32)
    - `gpu64`, `gpu32`, `pocl64`, `pocl32`: one correction on the GPU or on the CPU backend
      (POCL), compared with the default path
    - `profile32`: Float32 correction kernels on the GPU: tier breakdown, summary of the
      compiled code, and the device-side profile (time, threads, registers, memory per launch)
    - `breakdown32`: cost of the components of the integrand of the first tier (geometry,
      Lagrange basis, kernel and accumulation), with a fixed amount of work per pair
    - `rule32`, `rule32_fields`: reproducer of a crash of the CPU backend on x86, when a large
      struct is passed as a kernel argument (`rule32` is expected to crash there)
    Each stage prints a line before starting, so that a crash can be located. For the
    correction stages, the following environment variables can be set:
    - `INTI_DIAG_DIM=2`: ellipse (line elements) instead of a sphere
    - `INTI_DIAG_QUADS=1`: quadrilaterals instead of triangles (3D)
    - `INTI_DIAG_PRIVATE_BYTES=n`: per-thread memory budget of the first tier (default 4096)
    - `INTI_DIAG_SORT=0|1`: process the pairs sorted by cost (default: on GPUs only)
=#

using Inti, KernelAbstractions, AcceleratedKernels, ImplicitBVH, Gmsh, StaticArrays, SparseArrays, LinearAlgebra, Printf
const KA = KernelAbstractions
const Ext = Base.get_extension(Inti, :IntiKernelAbstractionsExt)

say(args...) = (println(args...); flush(stdout))

const BACKEND_NAME = get(ENV, "INTI_DIAG_BACKEND", Sys.isapple() ? "metal" : "cuda")
if BACKEND_NAME == "cuda"
    @eval using CUDA
    @eval begin
        gpu_backend() = CUDABackend()
        gpu_versioninfo() = CUDA.versioninfo()
        function compiled_code(f)
            io = IOBuffer()
            CUDA.@device_code_ptx io = io f()
            return String(take!(io))
        end
        gpu_profile(f) = CUDA.@profile trace = true f()
    end
elseif BACKEND_NAME == "metal"
    @eval using Metal
    @eval begin
        gpu_backend() = MetalBackend()
        gpu_versioninfo() = Metal.versioninfo()
        function compiled_code(f)
            io = IOBuffer()
            Metal.@device_code_air io = io f()
            return String(take!(io))
        end
        gpu_profile(f) = Metal.@profile trace = true f()
    end
else
    error("unknown INTI_DIAG_BACKEND = $BACKEND_NAME")
end

stage = isempty(ARGS) ? "kernels" : ARGS[1]
say("stage: ", stage, "   threads: ", Threads.nthreads(), "   GPU: ", BACKEND_NAME)
gpu_versioninfo()

## 1. kernels evaluated on the device vs on the host

@kernel function eval_kernel!(out, K, @Const(xs), @Const(ys))
    i = @index(Global, Linear)
    @inbounds out[i] = K(xs[i], ys[i])
end

function check_kernels()
    backend = gpu_backend()
    T = KA.supports_float64(backend) ? Float64 : Float32
    for dim in (2, 3)
        n = 10_000
        xs = [(coords = randn(SVector{dim, T}), normal = normalize(randn(SVector{dim, T}))) for _ in 1:n]
        ys = [(coords = x.coords + T(1.0e-3) * randn(SVector{dim, T}), normal = normalize(randn(SVector{dim, T}))) for x in xs]
        for op in (Inti.Laplace(; dim), Inti.Stokes{dim}(T(1))), K in (Inti.SingleLayerKernel(op), Inti.DoubleLayerKernel(op))
            ref = [K(xs[i], ys[i]) for i in 1:n]
            out = KA.allocate(backend, eltype(ref), n)
            eval_kernel!(backend)(out, K, Ext._to_device(backend, xs), Ext._to_device(backend, ys); ndrange = n)
            out = Array(out)
            err = maximum(i -> norm(out[i] - ref[i]) / norm(ref[i]), 1:n)
            @printf("  %dD %-24s max relative difference device vs host (%s): %.1e\n", dim, string(nameof(typeof(K)), " ", nameof(typeof(op))), T, err)
        end
    end
    return
end

## geometry

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

function sphere_quadrature(; meshsize, quads = false, order = 2, qorder = 4)
    Inti.clear_entities!()
    msh = try
        gmsh.initialize(String[], false)
        gmsh.option.setNumber("General.Verbosity", 2)
        gmsh.option.setNumber("Mesh.MeshSizeMax", meshsize)
        gmsh.option.setNumber("Mesh.MeshSizeMin", meshsize)
        quads && gmsh.option.setNumber("Mesh.RecombineAll", 1)
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

function geometry()
    dim = parse(Int, get(ENV, "INTI_DIAG_DIM", "3"))
    quads = get(ENV, "INTI_DIAG_QUADS", "0") == "1"
    h = dim == 2 ? 0.02 : 0.2
    Q = dim == 2 ? ellipse_quadrature(; meshsize = h) : sphere_quadrature(; meshsize = h, quads)
    X = [(coords = q.coords + s * δ * h * q.normal, normal = q.normal) for q in Q for δ in (1.0e-2, 0.1, 0.5) for s in (-1, 1)]
    return dim, quads, h, Q, X
end

# targets on `backend`, in the precision of the plan's centers
function device_targets(plan, X, backend)
    S = eltype(eltype(plan.centers))
    N = length(first(X).coords)
    return Ext._to_device(backend, [(coords = SVector{N, S}(x.coords), normal = SVector{N, S}(x.normal)) for x in X])
end

## 2. one correction with a given backend and precision, compared with the default path

function check_correction(backend, T)
    dim, quads, h, Q, X = geometry()
    private_bytes = parse(Int, get(ENV, "INTI_DIAG_PRIVATE_BYTES", string(Ext.PRIVATE_BYTES)))
    sort_by_cost = haskey(ENV, "INTI_DIAG_SORT") ? ENV["INTI_DIAG_SORT"] == "1" : !(backend isa typeof(CPU()))
    say("  dim = $dim, quads = $quads, private_bytes = $private_bytes, sort_by_cost = $sort_by_cost")
    K = Inti.SingleLayerKernel(Inti.Laplace(; dim))
    kw = T == Float32 ? (maxdist = 2h, rtol = 1.0e-5, atol = 1.0e-6) : (maxdist = 2h, rtol = 1.0e-6, atol = 0.0)
    say("  default path…")
    δref = Inti.adaptive_correction(Inti.IntegralOperator(K, X, Q); kw...)
    say("  plan…")
    plan = Inti.adaptive_correction_plan(K, Q; backend, T, kw...)
    Xd = device_targets(plan, X, backend)
    say("  search…")
    keys, npairs = Ext._nearfield_keys(plan, Xd)
    np = size(plan.el2qtags, 1)
    say("  integration ($npairs pairs, tiers $(Ext._tiers(plan, SVector{np, T}; private_bytes)))…")
    colval, nzval, nsub = Ext._nearfield_values(plan, Xd, keys, npairs; sort_by_cost, private_bytes)
    KA.synchronize(backend)
    say("  assembly…")
    rowptr = Ext._csr_rowptr(plan, keys, npairs, length(X))
    δ = SparseMatrixCSC(Ext.DeviceSparseMatrixCSR(length(X), length(Q), rowptr, colval, nzval))
    err = maximum(norm, nonzeros(δ - δref); init = 0.0) / maximum(norm, nonzeros(δref))
    ns = Array(nsub)
    @printf("  max entry error vs default path: %.1e   (nsub mean %.1f, max %d)\n", err, sum(abs, ns) / length(ns), maximum(abs, ns))
    return
end

## 3. profile of the Float32 correction kernels on the GPU

# Summary of the compiled code of the kernels whose name matches `pattern`: number of
# instructions, of double precision operations, and of accesses to per-thread (local/stack)
# memory, and its size. PTX (CUDA) and AIR (Metal, LLVM-based) are supported.
function code_report(code, pattern)
    if occursin(".entry", code) # PTX
        parts = split(code, r"\.entry "; keepempty = false)
        name_of = p -> first(split(p, '('))
        isinstr = l -> !isempty(l) && l ∉ ("{", "}") && !startswith(l, ".") && !startswith(l, "//") && !startswith(l, "\$") && !endswith(l, ":")
        isf64 = l -> occursin(".f64", l)
        isload = l -> occursin("ld.local", l)
        isstore = l -> occursin("st.local", l)
        stack = p -> (m = match(r"__local_depot\d+\[(\d+)\]", p); isnothing(m) ? 0 : parse(Int, m[1]))
    else # AIR (LLVM IR)
        parts = split(code, r"\ndefine "; keepempty = false)
        name_of = p -> (m = match(r"@([^\(]+)\(", p); isnothing(m) ? "" : m[1])
        isinstr = l -> !isempty(l) && !startswith(l, ";") && !endswith(l, ":") && l ∉ ("{", "}")
        isf64 = l -> occursin("double", l)
        isload = l -> false
        isstore = l -> false
        # stack arrays `alloca [N x type]` (other, struct-typed, allocas are not counted)
        scalar_bytes = Dict("i8" => 1, "i16" => 2, "i32" => 4, "i64" => 8, "half" => 2, "float" => 4, "double" => 8)
        stack = p -> sum(m -> parse(Int, m[1]) * scalar_bytes[m[2]], eachmatch(r"alloca \[(\d+) x (i8|i16|i32|i64|half|float|double)\]", p); init = 0)
    end
    for p in parts
        name = name_of(p)
        occursin(pattern, name) || continue
        instr = filter(isinstr, strip.(split(p, '\n')))
        f64 = filter(isf64, instr)
        ops = Dict{String, Int}()
        for l in f64
            op = first(split(l))
            ops[op] = get(ops, op, 0) + 1
        end
        say("    ", first(name, 70), "…")
        say(
            "      instructions: ", length(instr), ", with double precision: ", length(f64),
            ", local loads/stores: ", count(isload, instr), "/", count(isstore, instr),
            ", per-thread stack arrays (bytes): ", stack(p),
            ", allocations: ", count(l -> occursin(" alloca ", l), instr)
        )
        isempty(ops) || say("      double precision by opcode: ", sort(collect(ops); by = last, rev = true)[1:min(end, 8)])
    end
    return
end

function device_profile_lines(prof)
    s = sprint(show, MIME"text/plain"(), prof; context = :displaysize => (10_000, 400))
    lines = split(s, '\n')
    i = findfirst(l -> occursin("Device-side", l), lines)
    return isnothing(i) ? lines : lines[i:end]
end

function check_profile()
    backend = gpu_backend()
    sync() = KA.synchronize(backend)
    h = 0.2
    Q = sphere_quadrature(; meshsize = h, quads = get(ENV, "INTI_DIAG_QUADS", "0") == "1")
    X = [(coords = q.coords + s * δ * h * q.normal, normal = q.normal) for q in Q for δ in (1.0e-2, 0.1, 0.5) for s in (-1, 1)]
    for K in (Inti.SingleLayerKernel(Inti.Laplace(; dim = 3)), Inti.SingleLayerKernel(Inti.Stokes(; μ = 1.0, dim = 3)))
        kw = (maxdist = 2h, rtol = 1.0e-5, atol = 1.0e-6)
        plan = Inti.adaptive_correction_plan(K, Q; backend, T = Float32, kw...)
        Xd = device_targets(plan, X, backend)
        say("\n  ", nameof(typeof(K)), " ", nameof(typeof(K.op)), ", Float32 integration, targets in ", eltype(eltype(plan.centers)))
        keys, npairs = Ext._nearfield_keys(plan, Xd)
        # compiled code (captured while compiling, at the first call)
        code = compiled_code(() -> (Ext._nearfield_values(plan, Xd, keys, npairs); sync()))
        code_report(code, r"correction_(private|scratch|coop)")
        # tiers: pairs converging in each tier, time with all tiers and with the first only
        _, nzval, nsub = Ext._nearfield_values(plan, Xd, keys, npairs)
        np = size(plan.el2qtags, 1)
        caps = Ext._tiers(plan, SVector{np, eltype(nzval)})
        ns = Array(nsub)
        counts = [count(s -> (i == 1 ? -1 : caps[i - 1]) < s ≤ caps[i], ns) for i in eachindex(caps)]
        t_all = minimum(_ -> @elapsed((Ext._nearfield_values(plan, Xd, keys, npairs); sync())), 1:3)
        plan1 = Inti.adaptive_correction_plan(K, Q; backend, T = Float32, kw..., maxsubdiv = caps[1])
        Ext._nearfield_values(plan1, Xd, keys, npairs)
        t_1 = minimum(_ -> @elapsed((Ext._nearfield_values(plan1, Xd, keys, npairs); sync())), 1:3)
        say("    $npairs pairs; tiers (max subdivisions) $caps; pairs converging per tier $counts")
        @printf("    time: all tiers %.1f ms, first tier only %.1f ms (later tiers %.0f%%)\n", 1.0e3 * t_all, 1.0e3 * t_1, 100 * (t_all - t_1) / t_all)
        # device-side profile
        prof = gpu_profile(() -> (Ext._nearfield_values(plan, Xd, keys, npairs); sync()))
        foreach(l -> say(first(l, 220)), device_profile_lines(prof))
    end
    return
end

## 3b. cost breakdown of the integrand: every pair applies the rule on the reference element and
## on its children (a fixed number of evaluations), with the integrand reduced to the geometry
## (element and jacobian), the geometry and the Lagrange basis, or the full integrand

@kernel function ablation_kernel!(out, @Const(keys), @Const(X), @Const(els), @Const(offsets), @Const(oris), K, L, rnodes, rwh, rwl, rdom, ::Val{V}) where {V}
    p = @index(Global, Linear)
    rule = Ext.EmbeddedRule(rnodes, rwh, rwl, rdom)
    T = eltype(rwh)
    @inbounds key = keys[p]
    i, n = Ext._pair_from_key(key)
    @inbounds el, ori = els[n], oris[n]
    @inbounds x = Ext._local_target(T, X[i], offsets[n])
    f = if V === :geom
        ŷ -> (jac = Inti.jacobian(el, ŷ); y = el(ŷ); SVector(Ext._integration_measure(jac) + y[1] + x.coords[1]))
    elseif V === :basis
        ŷ -> (jac = Inti.jacobian(el, ŷ); y = el(ŷ); L(ŷ) * (Ext._integration_measure(jac) + y[1] + x.coords[1]))
    elseif V === :wide
        # same as `basis`, times a (cheap) 3×3 matrix: as many outputs as the Stokes integrand
        ŷ -> (jac = Inti.jacobian(el, ŷ); y = el(ŷ); M = (Ext._integration_measure(jac) + y[1] + x.coords[1]) * SMatrix{3, 3, T}(1, 2, 3, 4, 5, 6, 7, 8, 9); map(l -> M * l, L(ŷ)))
    else
        first(Ext._pair_integrand(key, X, els, offsets, oris, K, L, T))
    end
    dom = rule.domain
    I, E = Ext._apply_rule(f, rule, dom)
    children = Inti.HAdaptiveIntegration.default_subdivision(dom)(dom)
    for c in 1:length(children)
        Ic, Ec = Ext._apply_rule(f, rule, children[c])
        I += Ic
        E += Ec
    end
    @inbounds out[p] = norm(I) + E
end

# Same as `_apply_rule`, but one Lagrange basis function (output block) at a time: the integrand
# `fk(ŷ, k)` returns the block `k`, and the squared error is accumulated block by block.
@inline function apply_blocked(fk, rule, dom, ::Val{NB}) where {NB}
    Φ, μ = Inti.HAdaptiveIntegration.map_from_reference(dom)
    nodes, wh, wl = rule.nodes, rule.weights_high, rule.weights_low
    H, Lo = length(wh), length(wl)
    blocks = ntuple(Val(NB)) do k
        @inbounds begin
            v = fk(Φ(nodes[1]), k)
            Il = wl[1] * v
            Ih = wh[1] * v
            for i in 2:Lo
                v = fk(Φ(nodes[i]), k)
                Il += wl[i] * v
                Ih += wh[i] * v
            end
            for i in (Lo + 1):H
                Ih += wh[i] * fk(Φ(nodes[i]), k)
            end
        end
        (μ * Ih, sum(abs2, Ih - Il))
    end
    I = SVector(map(first, blocks))
    E2 = sum(map(last, blocks))
    return I, μ * sqrt(E2)
end

@kernel function blocked_kernel!(out, @Const(keys), @Const(X), @Const(els), @Const(offsets), @Const(oris), K, L, rnodes, rwh, rwl, rdom, ::Val{NB}) where {NB}
    p = @index(Global, Linear)
    rule = Ext.EmbeddedRule(rnodes, rwh, rwl, rdom)
    T = eltype(rwh)
    @inbounds key = keys[p]
    i, n = Ext._pair_from_key(key)
    @inbounds el, ori = els[n], oris[n]
    @inbounds x = Ext._local_target(T, X[i], offsets[n])
    fk = (ŷ, k) -> begin
        y = el(ŷ)
        jac = Inti.jacobian(el, ŷ)
        ν = Inti._normal(jac, ori)
        τ′ = Ext._integration_measure(jac)
        M = K(x, (coords = y, normal = ν))
        @inbounds Lk = L(ŷ)[k]
        M * (Lk * τ′)
    end
    dom = rule.domain
    I, E = apply_blocked(fk, rule, dom, Val(NB))
    children = Inti.HAdaptiveIntegration.default_subdivision(dom)(dom)
    for c in 1:length(children)
        Ic, Ec = apply_blocked(fk, rule, children[c], Val(NB))
        I += Ic
        E += Ec
    end
    @inbounds out[p] = norm(I) + E
end

function check_breakdown()
    backend = gpu_backend()
    sync() = KA.synchronize(backend)
    h = 0.2
    Q = sphere_quadrature(; meshsize = h, quads = get(ENV, "INTI_DIAG_QUADS", "0") == "1")
    X = [(coords = q.coords + s * δ * h * q.normal, normal = q.normal) for q in Q for δ in (1.0e-2, 0.1, 0.5) for s in (-1, 1)]
    for K in (Inti.SingleLayerKernel(Inti.Laplace(; dim = 3)), Inti.SingleLayerKernel(Inti.Stokes(; μ = 1.0, dim = 3)))
        plan = Inti.adaptive_correction_plan(K, Q; backend, T = Float32, maxdist = 2h, rtol = 1.0e-5, atol = 1.0e-6)
        Xd = device_targets(plan, X, backend)
        keys, npairs = Ext._nearfield_keys(plan, Xd)
        H = length(plan.rule.weights_high)
        nevals = H * (1 + Ext._nchildren(typeof(plan.rule.domain)))
        out = KA.allocate(backend, Float32, npairs)
        say("\n  ", nameof(typeof(K)), " ", nameof(typeof(K.op)), ": $npairs pairs, $nevals evaluations per pair")
        for V in (:geom, :basis, :wide, :full)
            run() = (ablation_kernel!(backend)(out, keys, Xd, plan.els, plan.offsets, plan.oris, plan.kernel, plan.basis, Ext._rule_fields(plan.rule)..., Val(V); ndrange = npairs); sync())
            run()
            t = minimum(_ -> @elapsed(run()), 1:3)
            @printf("    %-7s %8.2f ms   (%.2f ns per evaluation)\n", V, 1.0e3 * t, 1.0e9 * t / (npairs * nevals))
        end
        NB = size(plan.el2qtags, 1)
        runb() = (blocked_kernel!(backend)(out, keys, Xd, plan.els, plan.offsets, plan.oris, plan.kernel, plan.basis, Ext._rule_fields(plan.rule)..., Val(NB); ndrange = npairs); sync())
        runb()
        t = minimum(_ -> @elapsed(runb()), 1:3)
        @printf("    %-7s %8.2f ms   (full integrand, one basis function at a time)\n", :blocked, 1.0e3 * t)
    end
    return
end

## 4. reproducer of the crash of the CPU backend (POCL) on x86: the adaptive integrator on a
## Float32 triangle, with the rule passed as one struct (`rule32`, crashes) or field by field
## (`rule32_fields`, works)

const f_simple = u -> SVector(1 / ((u[1] - 0.3f0)^2 + (u[2] - 0.2f0)^2 + 1.0f-3), u[1] * u[2])

@kernel function rule_kernel!(out, rule)
    i = @index(Global, Linear)
    I, nsub, converged = Ext._adaptive_integrate(f_simple, rule, 1.0f-5, 0.0f0, Val(16))
    @inbounds out[i] = I[1] + nsub
end

@kernel function rule_fields_kernel!(out, nodes, wh, wl, dom)
    i = @index(Global, Linear)
    rule = Ext.EmbeddedRule(nodes, wh, wl, dom)
    I, nsub, converged = Ext._adaptive_integrate(f_simple, rule, 1.0f-5, 0.0f0, Val(16))
    @inbounds out[i] = I[1] + nsub
end

function check_rule(fields)
    rule = Ext._embedded_rule(Inti.ReferenceTriangle(), Float32)
    out = KA.allocate(CPU(), Float32, 10_000)
    if fields
        rule_fields_kernel!(CPU())(out, rule.nodes, rule.weights_high, rule.weights_low, rule.domain; ndrange = length(out))
    else
        rule_kernel!(CPU())(out, rule; ndrange = length(out))
    end
    KA.synchronize(CPU())
    return say("  ok, sum = ", sum(out))
end

if stage == "kernels"
    check_kernels()
elseif stage == "gpu64"
    check_correction(gpu_backend(), Float64)
elseif stage == "gpu32"
    check_correction(gpu_backend(), Float32)
elseif stage == "pocl64"
    check_correction(CPU(), Float64)
elseif stage == "pocl32"
    check_correction(CPU(), Float32)
elseif stage == "profile32"
    check_profile()
elseif stage == "breakdown32"
    check_breakdown()
elseif stage == "rule32"
    check_rule(false)
elseif stage == "rule32_fields"
    check_rule(true)
else
    error("unknown stage $stage")
end
say("done")
