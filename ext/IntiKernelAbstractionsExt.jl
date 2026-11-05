module IntiKernelAbstractionsExt

import Inti
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index, @Const
using LinearAlgebra
using SparseArrays
using StaticArrays

const HAdaptiveIntegration = Inti.HAdaptiveIntegration

function __init__()
    return @debug "Loading Inti.jl KernelAbstractions extension"
end

# Requested relative tolerances below `RTOL_FLOOR_FACTOR * eps(T)` are rejected a priori:
# below this level the embedded error estimate is dominated by rounding errors, so the
# tolerance cannot be certified. For `Float32` this is ≈ 7.6e-6. This condition is necessary
# but not sufficient, since rounding errors in the integrand depend on the conditioning of
# each interaction; tolerances that are not met are reported a posteriori.
const RTOL_FLOOR_FACTOR = 64

const DEFAULT_MAXSUBDIV = 256

function Inti._adaptive_correction_ka(
        iop,
        maxdist,
        rtol,
        atol,
        backend,
        ::Type{T},
        threads;
        maxsubdiv = DEFAULT_MAXSUBDIV,
        kwargs...,
    ) where {T}
    isempty(kwargs) || throw(
        ArgumentError(
            "unsupported keyword arguments for the KernelAbstractions backend: $(keys(kwargs))",
        ),
    )
    prep = _ka_to_device(_ka_host_prepare(iop, maxdist, rtol, atol, backend, T, maxsubdiv, threads))
    out = _ka_allocate_output(prep)
    _ka_nearfield!(out, prep)
    return _ka_assemble(prep, _ka_to_host(out))
end

function _check_precision(backend, ::Type{T}, rtol, atol) where {T}
    if T == Float64 && !KA.supports_float64(backend)
        throw(ArgumentError("backend $backend does not support Float64; use e.g. `T = Float32`"))
    end
    floor = RTOL_FLOOR_FACTOR * eps(T)
    # a pure absolute tolerance (`rtol = 0`) cannot be checked a priori; it is checked a
    # posteriori by the convergence flags of the kernel
    if !(iszero(rtol) && atol > 0) && rtol < floor
        msg = "rtol = $rtol is not achievable in $T (minimum supported rtol is $floor)"
        throw(ArgumentError(msg))
    end
    return
end

"""
    _ka_host_prepare(iop, maxdist, rtol, atol, backend, T, maxsubdiv, threads)

Build the list of nearly-singular (target, element) interactions, and all data needed by the
kernel converted to the floating point type `T`, on the host. See [`_ka_to_device`](@ref) for
the transfer to `backend`.

When possible (see [`_translate`](@ref)), each element, its quadrature nodes, and the targets
paired with it are expressed in coordinates centered at the element. The translation is
computed in the precision of the mesh before rounding to `T`, so that the rounding error in
`x - y` is of order `eps(T) * h` (with `h` the element size) instead of `eps(T) * |x|`.
"""
function _ka_host_prepare(iop, maxdist, rtol, atol, backend, ::Type{T}, maxsubdiv, threads) where {T}
    _check_precision(backend, T, rtol, atol)
    X, Y, K = Inti.target(iop), Inti.source(iop), Inti.kernel(iop)
    msh = Inti.mesh(Y)
    Kd = _convert_kernel(K, T)
    isbits(Kd) || throw(ArgumentError("kernel $K is not isbits"))
    dict_near = Inti.near_interaction_list([Inti.coords(x) for x in X], msh; tol = maxdist)
    # type-level flags, so that the node types below are inferred
    xnormal = Val(_has_normals(X))
    ynormal = Val(_has_normals(Y))
    rule = _embedded_rule(T)
    etypes = map(collect(Inti.element_types(msh))) do E
        Inti.reference_domain(E) isa Inti.ReferenceLine || throw(
            ArgumentError("only line elements are supported by the KernelAbstractions backend"),
        )
        els = Inti.elements(msh, E)
        oris = Inti.orientation(msh, E)
        el2qtags = Inti.etype2qtags(Y, E)
        pairs = _nearfield_pairs(X, Y, el2qtags, dict_near[E], maxdist, threads)
        qrule = Inti.quadrature_rule(Y, E)
        L = _lagrange_basis(qrule, T)
        # element centers (zero if no translation is performed)
        centers = map(el -> _translate(K, el) ? el(SVector(0.5)) : zero(el(SVector(0.5))), els)
        els_loc = [_convert_element(T, els[n], centers[n]) for n in eachindex(els)]
        ynodes = [_device_node(T, Y[el2qtags[k, n]], centers[n], ynormal) for k in axes(el2qtags, 1), n in axes(el2qtags, 2)]
        wnodes = [T(Inti.weight(Y[j])) for j in el2qtags]
        pair_el = [n for (_, n) in pairs]
        pair_x = [_device_node(T, X[i], centers[n], xnormal) for (i, n) in pairs]
        return (
            pairs = pairs,
            el2qtags = el2qtags,
            pair_el = pair_el,
            pair_x = pair_x,
            els = els_loc,
            oris = Int32.(oris),
            ynodes = ynodes,
            wnodes = wnodes,
            L = L,
        )
    end
    Ex = eltype(first(etypes).pair_x)
    Ey = eltype(first(etypes).ynodes)
    VT = Inti.return_type(Kd, Ex, Ey)
    return (
        backend = backend,
        size = size(iop),
        K = Kd,
        VT = VT,
        rule = rule,
        rtol = T(rtol),
        atol = T(atol),
        maxsubdiv = Val(maxsubdiv),
        etypes = etypes,
    )
end

"""
    _translate(K, el)

Whether the interaction of kernel `K` with element `el` can be computed in coordinates
centered at the element. This requires `K(x, y)` to depend on `x` and `y` only through
`x - y`, and an element type that can be translated.
"""
_translate(K::Inti.AbstractKernel, el) = _translation_invariant(K.op) && el isa Inti.LagrangeElement
_translate(K, el) = false
_translation_invariant(::Union{Inti.Laplace, Inti.Stokes}) = true
_translation_invariant(op) = false

"""
    _ka_to_device(prep)

Transfer the arrays of `prep` (see [`_ka_host_prepare`](@ref)) to `prep.backend`.
"""
function _ka_to_device(prep)
    backend = prep.backend
    etypes = map(prep.etypes) do et
        return merge(
            et,
            (
                pair_el = _to_device(backend, et.pair_el),
                pair_x = _to_device(backend, et.pair_x),
                els = _to_device(backend, et.els),
                oris = _to_device(backend, et.oris),
                ynodes = _to_device(backend, et.ynodes),
                wnodes = _to_device(backend, et.wnodes),
            ),
        )
    end
    KA.synchronize(backend)
    return merge(prep, (etypes = etypes,))
end

_to_device(backend, x::AbstractArray) = copyto!(KA.allocate(backend, eltype(x), size(x)), x)

# Same filter as in `Inti._adaptive_correction_etype!`: a target `i` is paired with element
# `n` if the closest quadrature node of `n` is within `maxdist`. Singular interactions (the
# target coincides with a quadrature node of `n`) are not supported.
function _nearfield_pairs(X, Y, el2qtags, nearlist, maxdist, threads)
    nel = size(el2qtags, 2)
    pairs_per_el = Vector{Vector{NTuple{2, Int32}}}(undef, nel)
    nsing = Threads.Atomic{Int}(0)
    Inti.@maybe_threads threads for n in 1:nel
        jglob = view(el2qtags, :, n)
        pairs_n = NTuple{2, Int32}[]
        for i in nearlist[n]
            xi = Inti.coords(X[i])
            dmin = minimum(j -> norm(xi - Inti.coords(Y[j])), jglob)
            dmin > maxdist && continue
            if iszero(dmin)
                Threads.atomic_add!(nsing, 1)
                continue
            end
            push!(pairs_n, (Int32(i), Int32(n)))
        end
        pairs_per_el[n] = pairs_n
    end
    if nsing[] > 0
        msg = """the KernelAbstractions backend supports only nearly-singular corrections,
        but $(nsing[]) target(s) coincide with source quadrature nodes. Use the default
        (`backend = nothing`) path instead."""
        throw(ArgumentError(msg))
    end
    # note: `reduce(vcat, ...)` without `init` uses an efficient specialization
    return isempty(pairs_per_el) ? NTuple{2, Int32}[] : reduce(vcat, pairs_per_el)
end

function _has_normals(X)
    normals = map(_normal_or_nothing, X)
    hasnormal = all(!isnothing, normals)
    if !hasnormal && any(!isnothing, normals)
        throw(ArgumentError("either all or none of the points must have a normal vector"))
    end
    return hasnormal
end

_normal_or_nothing(x::T) where {T} = hasfield(T, :normal) ? getfield(x, :normal) : nothing

# isbits `(coords, normal)` representation of `x - c` in precision `T`
function _device_node(::Type{T}, x, c, ::Val{hasnormal}) where {T, hasnormal}
    xc = Inti.coords(x) - c
    N = length(xc)
    nx = hasnormal ? SVector{N, T}(_normal_or_nothing(x)) : nothing
    return (coords = SVector{N, T}(xc), normal = nx)
end

# element translated by `-c` and converted to precision `T`
function _convert_element(::Type{T}, el::Inti.LagrangeElement{D, Np, SVector{N, S}}, c) where {T, D, Np, N, S}
    return Inti.LagrangeElement{D}(map(v -> SVector{N, T}(v - c), Inti.vals(el)))
end
function _convert_element(::Type{T}, el, c) where {T}
    E = typeof(el)
    iszero(c) || error("translation of element type $E is not supported")
    if !isbitstype(E) || eltype(Inti.return_type(el)) != T
        msg = "element type $E is not supported by the KernelAbstractions backend with T = $T"
        throw(ArgumentError(msg))
    end
    return el
end

_convert_kernel(K::Inti.AbstractKernel, ::Type{T}) where {T} =
    typeof(K).name.wrapper(_convert_op(K.op, T))
_convert_op(op::Inti.Laplace, ::Type{T}) where {T} = op
_convert_op(op::Inti.Stokes{N}, ::Type{T}) where {N, T} = Inti.Stokes{N}(T(op.μ))
function _convert_op(op, ::Type{T}) where {T}
    T == Float64 && return op
    throw(ArgumentError("conversion of $op to $T is not implemented"))
end

"""
    struct EmbeddedRule1D{H,L,T}

Isbits version of the embedded cubature used by `Inti.adaptive_quadrature(ReferenceLine())`
(i.e. the default rule of `HAdaptiveIntegration` for an `Orthotope{1}`). The low order rule
uses the first `L` of the `H` nodes.
"""
struct EmbeddedRule1D{H, L, T}
    nodes::SVector{H, T}
    weights_high::SVector{H, T}
    weights_low::SVector{L, T}
end

function _embedded_rule(::Type{T}) where {T}
    seg = HAdaptiveIntegration.Orthotope(zero(T), one(T))
    ec = HAdaptiveIntegration.default_rule(seg)
    H, L = length(ec.weights_high), length(ec.weights_low)
    return EmbeddedRule1D(
        SVector{H, T}(map(first, ec.nodes)),
        SVector{H, T}(ec.weights_high),
        SVector{L, T}(ec.weights_low),
    )
end

# Integrate `f` over `[a,b]` using `rule`. Mirrors `(::EmbeddedCubature)(f, ::Orthotope)` in
# HAdaptiveIntegration, including the order of the operations.
@inline function _apply_rule(f, rule::EmbeddedRule1D{H, L}, a, b) where {H, L}
    diff = b - a
    @inbounds begin
        v = f(SVector(a + rule.nodes[1] * diff))
        Il = rule.weights_low[1] * v
        Ih = rule.weights_high[1] * v
        for i in 2:L
            v = f(SVector(a + rule.nodes[i] * diff))
            Il += rule.weights_low[i] * v
            Ih += rule.weights_high[i] * v
        end
        for i in (L + 1):H
            Ih += rule.weights_high[i] * f(SVector(a + rule.nodes[i] * diff))
        end
    end
    return diff * Ih, diff * norm(Ih - Il)
end

"""
    _adaptive_integrate(f, rule, rtol, atol, ::Val{MAXSUBDIV})

Globally adaptive integration of `f` over `[0,1]`, reproducing `HAdaptiveIntegration.integrate`
on an `Orthotope{1}`: the panel with the largest error estimate is bisected until `E ≤ atol` or
`E ≤ rtol * norm(I)`, or until `MAXSUBDIV` subdivisions have been performed. The heap of
HAdaptiveIntegration is replaced by fixed-size arrays and a linear search, so that the function
can be called inside a GPU kernel.

Return `I, nsubdiv, converged`.
"""
@inline function _adaptive_integrate(f, rule::EmbeddedRule1D, rtol, atol, ::Val{MAXSUBDIV}) where {MAXSUBDIV}
    S = eltype(rule.nodes)
    a₀, b₀ = zero(S), one(S)
    I, E = _apply_rule(f, rule, a₀, b₀)
    CAP = MAXSUBDIV + 1 # each subdivision adds one panel
    as = MVector{CAP, S}(undef)
    bs = MVector{CAP, S}(undef)
    Is = MVector{CAP, typeof(I)}(undef)
    Es = MVector{CAP, typeof(E)}(undef)
    @inbounds as[1], bs[1], Is[1], Es[1] = a₀, b₀, I, E
    npanels = 1
    nsubdiv = 0
    converged = (E ≤ atol) || (E ≤ rtol * norm(I))
    @inbounds while !converged && nsubdiv < MAXSUBDIV
        # find the panel with the largest error
        k = 1
        Emax = Es[1]
        for m in 2:npanels
            if Es[m] > Emax
                k, Emax = m, Es[m]
            end
        end
        a, b = as[k], bs[k]
        I -= Is[k]
        E -= Es[k]
        mid = (a + b) / 2
        # children in the same order as `HAdaptiveIntegration.subdivide_orthotope`
        I₁, E₁ = _apply_rule(f, rule, mid, b)
        I += I₁
        E += E₁
        I₂, E₂ = _apply_rule(f, rule, a, mid)
        I += I₂
        E += E₂
        as[k], bs[k], Is[k], Es[k] = mid, b, I₁, E₁
        npanels += 1
        as[npanels], bs[npanels], Is[npanels], Es[npanels] = a, mid, I₂, E₂
        nsubdiv += 1
        converged = (E ≤ atol) || (E ≤ rtol * norm(I))
    end
    return I, nsubdiv, converged
end

"""
    struct LagrangeBasis1D{N,M}

Isbits version of `Inti.lagrange_basis(qrule)` for a quadrature rule on `ReferenceLine`.
"""
struct LagrangeBasis1D{N, M}
    C::M
end

@inline (L::LagrangeBasis1D{N})(x) where {N} = L.C * _line_monomials(Val(N), x[1])

# monomials `1, t, …, t^(N-1)` computed by recurrence (`Float32 ^ Int` is computed in
# `Float64` by Base, which is not supported by some GPU backends)
@inline function _line_monomials(::Val{N}, t::T) where {N, T}
    m = MVector{N, T}(undef)
    @inbounds m[1] = one(T)
    @inbounds for i in 2:N
        m[i] = m[i - 1] * t
    end
    return SVector(m)
end

function _lagrange_basis(qrule::Inti.ReferenceQuadrature{Inti.ReferenceLine}, ::Type{T}) where {T}
    N = length(qrule)
    nodes = Inti.qcoords(qrule)
    V = reduce(hcat, [_line_monomials(Val(N), Float64(x[1])) for x in nodes])
    C = SMatrix{N, N, T}(pinv(V))
    return LagrangeBasis1D{N, typeof(C)}(C)
end

# Same as `Inti._integration_measure`, but without the warning (string interpolation is not
# supported on GPUs).
@inline function _integration_measure(jac::SMatrix{M, N}) where {M, N}
    if M == N
        return abs(det(jac))
    else
        g = det(transpose(jac) * jac)
        return sqrt(max(g, zero(g)))
    end
end

@kernel function _nearfield_correction_kernel!(
        V,
        nsub,
        @Const(pair_el),
        @Const(pair_x),
        @Const(els),
        @Const(oris),
        @Const(ynodes),
        @Const(wnodes),
        K,
        L,
        rule,
        rtol,
        atol,
        maxsubdiv,
    )
    p = @index(Global, Linear)
    @inbounds begin
        n = pair_el[p]
        x = pair_x[p]
        el = els[n]
        ori = oris[n]
        integrand = (ŷ) -> begin
            y = el(ŷ)
            jac = Inti.jacobian(el, ŷ)
            ν = Inti._normal(jac, ori)
            τ′ = _integration_measure(jac)
            M = K(x, (coords = y, normal = ν))
            v = L(ŷ)
            map(v -> M * v, v) * τ′
        end
        W, nb, converged = _adaptive_integrate(integrand, rule, rtol, atol, maxsubdiv)
        for k in 1:length(W)
            V[k, p] = W[k] - K(x, ynodes[k, n]) * wnodes[k, n]
        end
        # number of subdivisions, or `-nb - 1` (always negative) if the tolerance was not met
        nsub[p] = converged ? Int32(nb) : -Int32(nb) - Int32(1)
    end
end

function _ka_allocate_output(prep)
    return map(prep.etypes) do et
        np = size(et.el2qtags, 1)
        npairs = length(et.pairs)
        V = KA.allocate(prep.backend, prep.VT, np, npairs)
        nsub = KA.allocate(prep.backend, Int32, npairs)
        return (V = V, nsub = nsub)
    end
end

"""
    _ka_nearfield!(out, prep; workgroupsize = nothing)

Launch the nearly-singular correction kernels. All inputs and outputs live on
`prep.backend`; the function returns after the kernels have completed. If `workgroupsize`
is `nothing`, the backend chooses it.
"""
function _ka_nearfield!(out, prep; workgroupsize = nothing)
    backend = prep.backend
    kernel! = if isnothing(workgroupsize)
        _nearfield_correction_kernel!(backend)
    else
        _nearfield_correction_kernel!(backend, workgroupsize)
    end
    for (et, o) in zip(prep.etypes, out)
        npairs = length(et.pairs)
        npairs == 0 && continue
        kernel!(
            o.V,
            o.nsub,
            et.pair_el,
            et.pair_x,
            et.els,
            et.oris,
            et.ynodes,
            et.wnodes,
            prep.K,
            et.L,
            prep.rule,
            prep.rtol,
            prep.atol,
            prep.maxsubdiv;
            ndrange = npairs,
        )
    end
    KA.synchronize(backend)
    return out
end

"""
    _ka_to_host(out)

Transfer the output of [`_ka_nearfield!`](@ref) to the host.
"""
_ka_to_host(out) = map(o -> (V = Array(o.V), nsub = Array(o.nsub)), out)

"""
    _ka_assemble(prep, out)

Check that all interactions in `out` (on the host) converged, and assemble the sparse
correction matrix.
"""
function _ka_assemble(prep, out)
    nfail = sum(o -> count(<(0), o.nsub), out; init = 0)
    if nfail > 0
        msg = """adaptive integration did not reach the requested tolerance for $nfail
        interaction(s) within `maxsubdiv = $(_val(prep.maxsubdiv))` subdivisions. Increase
        `maxsubdiv`, loosen the tolerance, or use a higher precision `T`. Note that a purely
        relative tolerance may not be attainable in precision `T` when an integral is small
        compared to its integrand (e.g. double-layer kernels); consider setting `atol`."""
        error(msg)
    end
    nnz = sum(o -> length(o.V), out; init = 0)
    Is, Js, Vs = Vector{Int}(undef, nnz), Vector{Int}(undef, nnz), Vector{prep.VT}(undef, nnz)
    idx = 0
    for (et, o) in zip(prep.etypes, out)
        for (p, (i, n)) in enumerate(et.pairs), k in axes(o.V, 1)
            idx += 1
            Is[idx], Js[idx], Vs[idx] = i, et.el2qtags[k, n], o.V[k, p]
        end
    end
    m, n = prep.size
    return sparse(Is, Js, Vs, m, n)
end

_val(::Val{N}) where {N} = N

end # module
