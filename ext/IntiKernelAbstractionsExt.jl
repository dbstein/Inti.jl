module IntiKernelAbstractionsExt

import Inti
import KernelAbstractions as KA
import AcceleratedKernels as AK
import ImplicitBVH
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

# Default bound on the number of subdivisions per interaction. The per-thread storage grows
# with it, which slows down the kernel on GPUs. In `Float32`, attainable tolerances were met
# with at most ~40 subdivisions in our tests (hypersingular kernel, targets at 1e-2 h from the
# boundary), and closer targets are limited by rounding (relative error ≈ eps(T) h / d) rather
# than by the number of subdivisions.
default_maxsubdiv(::Type{Float32}) = 64
default_maxsubdiv(::Type) = 256

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
    struct AdaptiveCorrectionPlan

Source data needed to compute nearly-singular corrections on a device; see
[`Inti.adaptive_correction_plan`](@ref). Element-wise data is stored in coordinates centered
at each element (when the kernel allows it, see [`_translate`](@ref)), with the centers
stored in the highest precision supported by the backend.
"""
struct AdaptiveCorrectionPlan{B, T, KT, LT, RT, MS, E, C, Y, W, Q, O, BV}
    backend::B
    kernel::KT
    basis::LT
    rule::RT
    maxdist::Float64
    rtol::T
    atol::T
    maxsubdiv::MS
    nsources::Int
    els::E      # elements (translated), in precision `T`
    offsets::C  # translation of each element (its center, or zero)
    centers::C  # element centers
    ynodes::Y   # `(coords, normal)` of the quadrature nodes (translated), `Np × nel`
    wnodes::W   # quadrature weights, `Np × nel`
    el2qtags::Q # global index of the quadrature nodes, `Np × nel`
    oris::O     # element orientations
    bvh::BV     # bounding volume hierarchy of the balls of radius ≈ `maxdist` around `centers`
end

function Base.show(io::IO, plan::AdaptiveCorrectionPlan{B, T}) where {B, T}
    nel = length(plan.els)
    return print(
        io, "AdaptiveCorrectionPlan on $(plan.backend) ($T): $nel elements, ",
        "$(plan.nsources) sources, maxdist = $(plan.maxdist)",
    )
end

_float_type(::AdaptiveCorrectionPlan{B, T}) where {B, T} = T
_val(::Val{N}) where {N} = N

function Inti._adaptive_correction_plan(K, Y, backend, ::Type{T}, maxdist, rtol, atol, maxsubdiv) where {T}
    _check_precision(backend, T, rtol, atol)
    maxsubdiv = something(maxsubdiv, default_maxsubdiv(T))
    Kd = _convert_kernel(K, T)
    isbits(Kd) || throw(ArgumentError("kernel $K is not isbits"))
    msh = Inti.mesh(Y)
    etypes = collect(Inti.element_types(msh))
    if length(etypes) != 1 || !(Inti.reference_domain(only(etypes)) isa Inti.ReferenceLine)
        msg = "only meshes with a single type of line elements are supported, got $etypes"
        throw(ArgumentError(msg))
    end
    E = only(etypes)
    els = Inti.elements(msh, E)
    el2qtags = Inti.etype2qtags(Y, E)
    L = _lagrange_basis(Inti.quadrature_rule(Y, E), T)
    # element centers and translations, in the highest precision available on the backend
    S = KA.supports_float64(backend) ? Float64 : T
    N = Inti.ambient_dimension(msh)
    centers = [SVector{N, S}(el(SVector(0.5))) for el in els]
    offsets = [_translate(K, els[n]) ? centers[n] : zero(centers[n]) for n in eachindex(els)]
    els_loc = [_convert_element(T, els[n], offsets[n]) for n in eachindex(els)]
    ynormal = Val(_has_normals(Y))
    ynodes = [
        _device_node(T, Y[el2qtags[k, n]], offsets[n], ynormal) for
            k in axes(el2qtags, 1), n in axes(el2qtags, 2)
    ]
    wnodes = [T(Inti.weight(Y[j])) for j in el2qtags]
    # BVH of the balls of radius `maxdist` around the centers. The radius is enlarged to
    # account for the rounding of the centers and targets to `T`; the exact criterion is
    # applied afterwards.
    cmax = maximum(c -> norm(c, Inf), centers)
    r = maxdist + 8 * eps(T) * (cmax + maxdist)
    balls = [_bsphere(T, c, r) for c in centers]
    bvh = ImplicitBVH.BVH(_to_device(backend, balls), ImplicitBVH.BBox{T})
    return AdaptiveCorrectionPlan(
        backend,
        Kd,
        L,
        _embedded_rule(T),
        Float64(maxdist),
        T(rtol),
        T(atol),
        Val(maxsubdiv),
        length(Y),
        _to_device(backend, els_loc),
        _to_device(backend, offsets),
        _to_device(backend, centers),
        _to_device(backend, ynodes),
        _to_device(backend, wnodes),
        _to_device(backend, Int32.(el2qtags)),
        _to_device(backend, Int32.(Inti.orientation(msh, E))),
        bvh,
    )
end

# bounding sphere (in 3D, as required by ImplicitBVH) of radius `r` centered at `x`
@inline function _bsphere(::Type{T}, x, r) where {T}
    if length(x) == 2
        return ImplicitBVH.BSphere{T}((T(x[1]), T(x[2]), zero(T)), T(r))
    else
        return ImplicitBVH.BSphere{T}((T(x[1]), T(x[2]), T(x[3])), T(r))
    end
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

_to_device(backend, x::AbstractArray) = copyto!(KA.allocate(backend, eltype(x), size(x)), x)

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

# Target `x` in coordinates centered at `c`, in precision `T`. The translation is computed in
# the highest of the precisions of `x` and `c`.
@inline function _local_target(::Type{T}, x, c) where {T}
    xc = Inti.coords(x)
    N = length(xc)
    R = promote_type(eltype(xc), eltype(c))
    coords = SVector{N, T}(SVector{N, R}(xc) - SVector{N, R}(c))
    nx = _normal_or_nothing(x)
    return (coords = coords, normal = isnothing(nx) ? nothing : SVector{N, T}(nx))
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

## Search of the nearly-singular interactions

# keys encoding a (target, element) pair, sorted by target first
@inline _pair_key(i, n) = (UInt64(i) << 32) | UInt64(n)
@inline _pair_from_key(key) = (Int32(key >> 32), Int32(key & 0xffffffff))
const INVALID_KEY = typemax(UInt64)

# Same criterion as `Inti.near_interaction_list` followed by the filter in
# `Inti._adaptive_correction_etype!`: the target `i` interacts with element `n` if it is within
# `maxdist` of the center of `n`, and if its closest quadrature node in `n` is within
# `maxdist`. A target whose distance to a quadrature node is below the rounding errors made
# in computing that distance (rounding of the target, of the translation, and of the node in
# local coordinates) is considered to coincide with it (singular interaction).
@kernel function _filter_candidates_kernel!(
        keys, nsing, @Const(candidates), @Const(X), @Const(offsets), @Const(centers),
        @Const(ynodes), maxdist,
    )
    p = @index(Global, Linear)
    @inbounds begin
        n, i = candidates[p]
        x = Inti.coords(X[i])
        c, o = centers[n], offsets[n]
        N = length(x)
        R = promote_type(eltype(x), eltype(c))
        valid = norm(SVector{N, R}(x) - SVector{N, R}(c)) ≤ maxdist
        xo = SVector{N, R}(x) - SVector{N, R}(o)
        dmin = typemax(R)
        singular = false
        for k in axes(ynodes, 1)
            yk = ynodes[k, n].coords
            d = norm(xo - SVector{N, R}(yk))
            tol = 2 * (eps(eltype(x)) * norm(x) + eps(R) * norm(o) + eps(eltype(yk)) * norm(yk))
            singular |= d ≤ tol
            dmin = min(dmin, d)
        end
        valid &= dmin ≤ maxdist
        singular &= valid
        if singular
            KA.@atomic nsing[1] += Int32(1)
        end
        keys[p] = (valid && !singular) ? _pair_key(i, n) : INVALID_KEY
    end
end

@kernel function _target_balls_kernel!(balls, @Const(X))
    i = @index(Global, Linear)
    @inbounds balls[i] = _bsphere(_float_type(eltype(balls)), Inti.coords(X[i]), 0)
end

_float_type(::Type{ImplicitBVH.BSphere{T}}) where {T} = T

"""
    _nearfield_keys(plan, X)

Return a sorted device vector of keys encoding the nearly-singular (target, element) pairs,
and the number `npairs` of such pairs (the first `npairs` keys).
"""
function _nearfield_keys(plan::AdaptiveCorrectionPlan, X)
    backend = plan.backend
    T = _float_type(plan)
    balls = KA.allocate(backend, ImplicitBVH.BSphere{T}, length(X))
    _target_balls_kernel!(backend)(balls, X; ndrange = length(X))
    bvh = ImplicitBVH.BVH(balls, ImplicitBVH.BBox{T})
    traversal = ImplicitBVH.traverse(plan.bvh, bvh)
    ncand = traversal.num_contacts
    keys = KA.allocate(backend, UInt64, ncand)
    ncand == 0 && return keys, 0
    nsing = KA.zeros(backend, Int32, 1)
    _filter_candidates_kernel!(backend)(
        keys, nsing, traversal.contacts, X, plan.offsets, plan.centers, plan.ynodes,
        eltype(eltype(plan.centers))(plan.maxdist); ndrange = ncand,
    )
    nsing = Array(nsing)[1]
    if nsing > 0
        msg = """the KernelAbstractions backend supports only nearly-singular corrections,
        but $nsing target(s) coincide with source quadrature nodes. Use the default
        (`backend = nothing`) path instead."""
        throw(ArgumentError(msg))
    end
    AK.sort!(keys)
    npairs = AK.mapreduce(k -> Int(k != INVALID_KEY), +, keys; init = 0)
    return keys, npairs
end

## Assembly

@kernel function _row_keys_kernel!(rowkeys)
    i = @index(Global, Linear)
    @inbounds rowkeys[i] = _pair_key(i, 0)
end

@kernel function _rowptr_kernel!(rowptr, @Const(first_pair), np)
    i = @index(Global, Linear)
    @inbounds rowptr[i] = Int32(np * (first_pair[i] - 1) + 1)
end

# Row pointers of the CSR matrix: the keys are sorted by target, and each pair contributes
# `np` consecutive entries.
function _csr_rowptr(plan::AdaptiveCorrectionPlan, keys, npairs, m)
    backend = plan.backend
    np = size(plan.el2qtags, 1)
    np * npairs < typemax(Int32) || error("number of nonzeros exceeds typemax(Int32)")
    rowkeys = KA.allocate(backend, UInt64, m + 1)
    _row_keys_kernel!(backend)(rowkeys; ndrange = m + 1)
    first_pair = KA.allocate(backend, Int, m + 1)
    AK.searchsortedfirst!(first_pair, view(keys, 1:npairs), rowkeys)
    rowptr = KA.allocate(backend, Int32, m + 1)
    _rowptr_kernel!(backend)(rowptr, first_pair, np; ndrange = m + 1)
    return rowptr
end

# Proxy for the cost of the adaptive integration of pair `p`: distance from the target to the
# closest quadrature node, relative to the extent of the element's quadrature nodes. The
# number of subdivisions grows as this ratio decreases.
@kernel function _pair_cost_kernel!(cost, @Const(keys), @Const(X), @Const(offsets), @Const(ynodes))
    p = @index(Global, Linear)
    @inbounds begin
        i, n = _pair_from_key(keys[p])
        T = eltype(cost)
        x = _local_target(T, X[i], offsets[n]).coords
        np = size(ynodes, 1)
        dmin = typemax(T)
        for k in 1:np
            dmin = min(dmin, norm(x - ynodes[k, n].coords))
        end
        cost[p] = dmin / norm(ynodes[np, n].coords - ynodes[1, n].coords)
    end
end

# Permutation grouping pairs of similar cost, so that the work-items executed in lockstep on a
# GPU (warps, SIMD groups) perform similar numbers of subdivisions.
function _cost_order(plan::AdaptiveCorrectionPlan, X, keys, npairs)
    backend = plan.backend
    cost = KA.allocate(backend, _float_type(plan), npairs)
    _pair_cost_kernel!(backend)(cost, keys, X, plan.offsets, plan.ynodes; ndrange = npairs)
    perm = KA.allocate(backend, Int32, npairs)
    AK.sortperm!(perm, cost)
    return perm
end

@kernel function _nearfield_correction_kernel!(
        nzval, colval, nsub, @Const(perm), @Const(keys), @Const(X), @Const(els),
        @Const(offsets), @Const(oris), @Const(ynodes), @Const(wnodes), @Const(el2qtags),
        K, L, rule, rtol, atol, maxsubdiv,
    )
    t = @index(Global, Linear)
    @inbounds begin
        p = isnothing(perm) ? t : Int(perm[t])
        i, n = _pair_from_key(keys[p])
        T = typeof(rtol)
        x = _local_target(T, X[i], offsets[n])
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
        np = length(W)
        for k in 1:np
            idx = np * (p - 1) + k
            nzval[idx] = W[k] - K(x, ynodes[k, n]) * wnodes[k, n]
            colval[idx] = el2qtags[k, n]
        end
        # number of subdivisions, or `-nb - 1` (always negative) if the tolerance was not met
        nsub[p] = converged ? Int32(nb) : -Int32(nb) - Int32(1)
    end
end

"""
    _nearfield_values(plan, X, keys, npairs; sort_by_cost, workgroupsize = nothing)

Compute the values (and column indices) of the CSR correction matrix, together with the
number of subdivisions used for each pair. If `sort_by_cost`, the pairs are processed in the
order given by [`_cost_order`](@ref). This is the default on GPUs, where it reduces the
divergence between work-items executed in lockstep; on the CPU backend the original order
(pairs of the same target and neighboring elements are adjacent) is more cache-friendly.
"""
function _nearfield_values(
        plan::AdaptiveCorrectionPlan, X, keys, npairs;
        sort_by_cost = !(plan.backend isa typeof(KA.CPU())), workgroupsize = nothing,
    )
    backend = plan.backend
    np = size(plan.el2qtags, 1)
    T = _float_type(plan)
    XT = Base.promote_op(_local_target, Type{T}, eltype(X), eltype(plan.offsets))
    VT = Inti.return_type(plan.kernel, XT, eltype(plan.ynodes))
    nzval = KA.allocate(backend, VT, np * npairs)
    colval = KA.allocate(backend, Int32, np * npairs)
    nsub = KA.allocate(backend, Int32, npairs)
    npairs == 0 && return colval, nzval, nsub
    kernel! = if isnothing(workgroupsize)
        _nearfield_correction_kernel!(backend)
    else
        _nearfield_correction_kernel!(backend, workgroupsize)
    end
    perm = sort_by_cost ? _cost_order(plan, X, keys, npairs) : nothing
    kernel!(
        nzval, colval, nsub, perm, keys, X, plan.els, plan.offsets, plan.oris, plan.ynodes,
        plan.wnodes, plan.el2qtags, plan.kernel, plan.basis, plan.rule, plan.rtol, plan.atol,
        plan.maxsubdiv; ndrange = npairs,
    )
    return colval, nzval, nsub
end

function _check_convergence(plan::AdaptiveCorrectionPlan, nsub)
    isempty(nsub) && return
    nfail = AK.mapreduce(s -> Int(s < 0), +, nsub; init = 0)
    if nfail > 0
        msg = """adaptive integration did not reach the requested tolerance for $nfail
        interaction(s) within `maxsubdiv = $(_val(plan.maxsubdiv))` subdivisions. Increase
        `maxsubdiv`, loosen the tolerance, or use a higher precision `T`. Note that a purely
        relative tolerance may not be attainable in precision `T` when an integral is small
        compared to its integrand (e.g. double-layer kernels); consider setting `atol`."""
        error(msg)
    end
    return
end

function Inti.adaptive_correction(plan::AdaptiveCorrectionPlan, X::AbstractVector)
    m = length(X)
    keys, npairs = _nearfield_keys(plan, X)
    rowptr = _csr_rowptr(plan, keys, npairs, m)
    colval, nzval, nsub = _nearfield_values(plan, X, keys, npairs)
    _check_convergence(plan, nsub)
    return DeviceSparseMatrixCSR(m, plan.nsources, rowptr, colval, nzval)
end

# convenience path used by `adaptive_correction(iop; backend)`: targets on the host, result
# returned as a `SparseMatrixCSC`
function Inti._adaptive_correction_host(plan::AdaptiveCorrectionPlan, X)
    S = eltype(eltype(plan.centers))
    xnormal = Val(_has_normals(X))
    Xd = _to_device(plan.backend, [_device_node(S, x, zero(Inti.coords(x)), xnormal) for x in X])
    return SparseMatrixCSC(Inti.adaptive_correction(plan, Xd))
end

## Sparse matrix in CSR format on a device

"""
    struct DeviceSparseMatrixCSR{Tv}

Sparse matrix in compressed sparse row format, with the arrays `rowptr`, `colval` and
`nzval` living on a device. The column indices within each row are sorted. Supports
`mul!(y, A, x[, α, β])` and `A * x` for vectors on the same device, and conversion to the
host with `SparseMatrixCSC(A)`.
"""
struct DeviceSparseMatrixCSR{Tv, P, C, V}
    m::Int
    n::Int
    rowptr::P
    colval::C
    nzval::V
    function DeviceSparseMatrixCSR(m, n, rowptr::P, colval::C, nzval::V) where {P, C, V}
        return new{eltype(nzval), P, C, V}(m, n, rowptr, colval, nzval)
    end
end

Base.size(A::DeviceSparseMatrixCSR) = (A.m, A.n)
Base.size(A::DeviceSparseMatrixCSR, d::Integer) = d == 1 ? A.m : (d == 2 ? A.n : 1)
Base.eltype(::DeviceSparseMatrixCSR{Tv}) where {Tv} = Tv
SparseArrays.nnz(A::DeviceSparseMatrixCSR) = length(A.nzval)
KA.get_backend(A::DeviceSparseMatrixCSR) = KA.get_backend(A.nzval)

function Base.show(io::IO, A::DeviceSparseMatrixCSR)
    return print(
        io, "$(A.m)×$(A.n) DeviceSparseMatrixCSR{$(eltype(A))} with $(nnz(A)) stored entries ",
        "on $(KA.get_backend(A))",
    )
end

@kernel function _csr_mul_kernel!(y, @Const(rowptr), @Const(colval), @Const(nzval), @Const(x), α, β)
    i = @index(Global, Linear)
    @inbounds begin
        acc = zero(eltype(y))
        for k in rowptr[i]:(rowptr[i + 1] - 1)
            acc += nzval[k] * x[colval[k]]
        end
        y[i] = iszero(β) ? α * acc : α * acc + β * y[i]
    end
end

function LinearAlgebra.mul!(y::AbstractVector, A::DeviceSparseMatrixCSR, x::AbstractVector, α::Number, β::Number)
    size(A, 2) == length(x) || throw(DimensionMismatch("A has $(size(A, 2)) columns, x has length $(length(x))"))
    size(A, 1) == length(y) || throw(DimensionMismatch("A has $(size(A, 1)) rows, y has length $(length(y))"))
    backend = KA.get_backend(A)
    size(A, 1) == 0 && return y
    _csr_mul_kernel!(backend)(y, A.rowptr, A.colval, A.nzval, x, α, β; ndrange = size(A, 1))
    return y
end

LinearAlgebra.mul!(y::AbstractVector, A::DeviceSparseMatrixCSR, x::AbstractVector) =
    mul!(y, A, x, true, false)

function Base.:*(A::DeviceSparseMatrixCSR, x::AbstractVector)
    Ty = Base.promote_op(*, eltype(A), eltype(x))
    y = KA.allocate(KA.get_backend(A), Ty, size(A, 1))
    return mul!(y, A, x)
end

function SparseArrays.SparseMatrixCSC(A::DeviceSparseMatrixCSR)
    rowptr, colval, nzval = Array(A.rowptr), Array(A.colval), Array(A.nzval)
    I = Vector{Int}(undef, length(nzval))
    for i in 1:A.m, k in rowptr[i]:(rowptr[i + 1] - 1)
        I[k] = i
    end
    return sparse(I, Int.(colval), nzval, A.m, A.n)
end

end # module
