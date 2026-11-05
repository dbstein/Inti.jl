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

# Default bound on the number of subdivisions per interaction, for line (2D) and surface (3D)
# elements. In our tests, at most ~130 (line elements, rtol = 1e-8) and ~950 (surface elements,
# rtol = 1e-6) subdivisions were needed for targets at 1e-2 h from the boundary. Since the
# integration is performed in tiers (see `_tiers`), a large bound only costs memory and time
# for the interactions that need it.
default_maxsubdiv(ref::Inti.ReferenceShape) = Inti.geometric_dimension(ref) == 1 ? 1024 : 4096

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
    Kd = _convert_kernel(K, T)
    isbits(Kd) || throw(ArgumentError("kernel $K is not isbits"))
    msh = Inti.mesh(Y)
    etypes = collect(Inti.element_types(msh))
    supported = Union{Inti.ReferenceLine, Inti.ReferenceTriangle, Inti.ReferenceSquare}
    if length(etypes) != 1 || !(Inti.reference_domain(only(etypes)) isa supported) ||
            Inti.geometric_dimension(Inti.reference_domain(only(etypes))) != Inti.ambient_dimension(msh) - 1
        msg = """only boundary meshes with a single type of line, triangle, or quadrilateral
        elements are supported, got $etypes"""
        throw(ArgumentError(msg))
    end
    E = only(etypes)
    maxsubdiv = something(maxsubdiv, default_maxsubdiv(Inti.reference_domain(E)))
    els = Inti.elements(msh, E)
    el2qtags = Inti.etype2qtags(Y, E)
    L = _lagrange_basis(Inti.quadrature_rule(Y, E), T)
    # element centers and translations, in the highest precision available on the backend
    S = KA.supports_float64(backend) ? Float64 : T
    N = Inti.ambient_dimension(msh)
    centers = [SVector{N, S}(Inti.center(el)) for el in els]
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
        _embedded_rule(Inti.reference_domain(E), T),
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
    struct EmbeddedRule{H,L,D,T,DOM}

Isbits version of the embedded cubature used by `Inti.adaptive_quadrature` on a reference
shape, i.e. the default rule of `HAdaptiveIntegration` for the corresponding domain
`domain`. The low order rule uses the first `L` of the `H` nodes.
"""
struct EmbeddedRule{H, L, D, T, DOM}
    nodes::SVector{H, SVector{D, T}}
    weights_high::SVector{H, T}
    weights_low::SVector{L, T}
    domain::DOM
end

# domains used by `Inti.adaptive_quadrature` for each reference shape, in precision `T`
_hadaptive_domain(::Inti.ReferenceLine, ::Type{T}) where {T} =
    HAdaptiveIntegration.Orthotope(zero(T), one(T))
_hadaptive_domain(::Inti.ReferenceTriangle, ::Type{T}) where {T} =
    HAdaptiveIntegration.Triangle((zero(T), zero(T)), (one(T), zero(T)), (zero(T), one(T)))
_hadaptive_domain(::Inti.ReferenceSquare, ::Type{T}) where {T} =
    HAdaptiveIntegration.Rectangle((zero(T), zero(T)), (one(T), one(T)))

function _embedded_rule(ref::Inti.ReferenceShape, ::Type{T}) where {T}
    domain = _hadaptive_domain(ref, T)
    ec = HAdaptiveIntegration.default_rule(domain)
    H, L, D = length(ec.weights_high), length(ec.weights_low), length(first(ec.nodes))
    return EmbeddedRule(
        SVector{H}(map(x -> SVector{D, T}(x), ec.nodes)),
        SVector{H, T}(ec.weights_high),
        SVector{L, T}(ec.weights_low),
        domain,
    )
end

# number of children created by `HAdaptiveIntegration.default_subdivision`
_nchildren(::Type{<:HAdaptiveIntegration.Orthotope{D}}) where {D} = 2^D
_nchildren(::Type{<:HAdaptiveIntegration.Simplex{2}}) = 4

# Integrate `f` over `domain` using `rule`. Mirrors `(::EmbeddedCubature)(f, domain)` in
# HAdaptiveIntegration, including the order of the operations.
@inline function _apply_rule(f, rule::EmbeddedRule{H, L}, domain) where {H, L}
    Φ, μ = HAdaptiveIntegration.map_from_reference(domain)
    @inbounds begin
        v = f(Φ(rule.nodes[1]))
        Il = rule.weights_low[1] * v
        Ih = rule.weights_high[1] * v
        for i in 2:L
            v = f(Φ(rule.nodes[i]))
            Il += rule.weights_low[i] * v
            Ih += rule.weights_high[i] * v
        end
        for i in (L + 1):H
            Ih += rule.weights_high[i] * f(Φ(rule.nodes[i]))
        end
    end
    return μ * Ih, μ * norm(Ih - Il)
end

# Whether the integral over each panel is stored (rather than recomputed when the panel is
# subdivided). Storing it saves one application of the rule per subdivision, but increases
# the per-thread memory, which slows down GPU kernels when the integrand has many components
# (e.g. Stokes kernels in 3D).
_store_integrals(::Type{I}, ::Type{T}) where {I, T} = sizeof(I) ≤ 16 * sizeof(T)

# number of panels after `maxsubdiv` subdivisions of a domain of type `DOM`
_capacity(::Type{DOM}, maxsubdiv) where {DOM} = 1 + (_nchildren(DOM) - 1) * maxsubdiv

"""
    _adaptive_integrate(f, rule, rtol, atol, ::Val{MAXSUBDIV})

Globally adaptive integration of `f` over `rule.domain`, reproducing
`HAdaptiveIntegration.integrate`, with at most `MAXSUBDIV` subdivisions and the panels stored
in fixed-size per-thread arrays. See [`_refine!`](@ref).

Return `I, nsubdiv, converged`.
"""
@inline function _adaptive_integrate(f, rule::EmbeddedRule, rtol, atol, ::Val{MAXSUBDIV}) where {MAXSUBDIV}
    S = eltype(eltype(rule.nodes))
    DOM = typeof(rule.domain)
    I, E = _apply_rule(f, rule, rule.domain)
    CAP = _capacity(DOM, MAXSUBDIV)
    STORE = _store_integrals(typeof(I), S)
    panels = MVector{CAP, DOM}(undef)
    Es = MVector{CAP, typeof(E)}(undef)
    Is = MVector{STORE ? CAP : 0, typeof(I)}(undef)
    return _refine!(panels, Es, Is, f, rule, I, E, rtol, atol, MAXSUBDIV, Val(STORE))
end

"""
    _refine!(panels, Es, Is, f, rule, I, E, rtol, atol, maxsubdiv, ::Val{STORE})

Refinement loop of [`_adaptive_integrate`](@ref), given the integral `I` and error estimate `E`
over `rule.domain`. The panel with the largest error estimate is subdivided (with
`HAdaptiveIntegration.default_subdivision`) until `E ≤ atol` or `E ≤ rtol * norm(I)`, or until
`maxsubdiv` subdivisions have been performed, exactly as in `HAdaptiveIntegration.integrate`
(whose heap is replaced by the arrays `panels`, `Es` and `Is`, and a linear search). If
`STORE`, the integral over each panel is stored in `Is`; otherwise it is recomputed when the
panel is subdivided (giving the same value, bit for bit).

The arrays (e.g. per-thread static arrays, or views of global device memory) must have room
for [`_capacity`](@ref) panels.
"""
@inline function _refine!(panels, Es, Is, f, rule::EmbeddedRule, I, E, rtol, atol, maxsubdiv, ::Val{STORE}) where {STORE}
    dom₀ = rule.domain
    NC = _nchildren(typeof(dom₀))
    subdivide = HAdaptiveIntegration.default_subdivision(dom₀)
    @inbounds panels[1], Es[1] = dom₀, E
    STORE && (@inbounds Is[1] = I)
    npanels = 1
    nsubdiv = 0
    converged = (E ≤ atol) || (E ≤ rtol * norm(I))
    @inbounds while !converged && nsubdiv < maxsubdiv
        # find the panel with the largest error
        k = 1
        Emax = Es[1]
        for m in 2:npanels
            if Es[m] > Emax
                k, Emax = m, Es[m]
            end
        end
        dom = panels[k]
        Iₖ = STORE ? Is[k] : first(_apply_rule(f, rule, dom))
        I -= Iₖ
        E -= Es[k]
        children = subdivide(dom)
        for c in 1:NC
            child = children[c]
            Ic, Ec = _apply_rule(f, rule, child)
            I += Ic
            E += Ec
            # the first child replaces its parent, the others are appended
            slot = c == 1 ? k : (npanels += 1)
            panels[slot], Es[slot] = child, Ec
            STORE && (Is[slot] = Ic)
        end
        nsubdiv += 1
        converged = (E ≤ atol) || (E ≤ rtol * norm(I))
    end
    return I, nsubdiv, converged
end

"""
    struct LagrangeBasis{K,E,M}

Isbits version of `Inti.lagrange_basis(qrule)`: `L(x) = C * m(x)`, where `m(x)` are the
monomials `x^e` for the exponents `e` in `exponents` (all of degree at most `K` in each
variable), and `C` is the pseudo-inverse of the Vandermonde matrix at the quadrature nodes.
"""
struct LagrangeBasis{K, E, M}
    exponents::E
    C::M
end

@inline (L::LagrangeBasis)(x) = L.C * _monomials(L, x)

@inline function _monomials(L::LagrangeBasis{K}, x::SVector{D}) where {K, D}
    p = ntuple(d -> _powers(Val(K + 1), x[d]), Val(D))
    return SVector(map(e -> prod(ntuple(d -> p[d][e[d] + 1], Val(D))), L.exponents))
end

# powers `1, t, …, t^(N-1)` computed by recurrence (`Float32 ^ Int` is computed in `Float64`
# by Base, which is not supported by some GPU backends)
@inline function _powers(::Val{N}, t::T) where {N, T}
    m = MVector{N, T}(undef)
    @inbounds m[1] = one(T)
    @inbounds for i in 2:N
        m[i] = m[i - 1] * t
    end
    return SVector(m)
end

# exponents of the monomials spanning the polynomial space of `qrule`
_exponents(::Inti.ReferenceLine, k) = [(i,) for i in 0:k]
_exponents(::Inti.ReferenceTriangle, k) = [(i, j) for j in 0:k for i in 0:k if i + j ≤ k]
_exponents(::Inti.ReferenceSquare, k) = [(i, j) for j in 0:k for i in 0:k]

function _lagrange_basis(qrule::Inti.ReferenceQuadrature, ::Type{T}) where {T}
    ref = Inti.domain(qrule)
    k = Inti.interpolation_order(qrule)
    exps = Tuple(_exponents(ref, k))
    nodes = Inti.qcoords(qrule)
    L64 = LagrangeBasis{k, typeof(exps), Nothing}(exps, nothing)
    V = reduce(hcat, [_monomials(L64, SVector{length(x), Float64}(x)) for x in nodes])
    C = SMatrix{size(V, 2), size(V, 1), T}(pinv(V))
    return LagrangeBasis{k, typeof(exps), typeof(C)}(exps, C)
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
        dmin, spread = typemax(T), zero(T)
        y₁ = ynodes[1, n].coords
        for k in axes(ynodes, 1)
            yk = ynodes[k, n].coords
            dmin = min(dmin, norm(x - yk))
            spread = max(spread, norm(yk - y₁))
        end
        cost[p] = dmin / spread
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

# Data of the pair encoded in `key`: target (in coordinates centered at the element), element,
# orientation, and element index.
@inline function _pair_data(key, X, els, offsets, oris, ::Type{T}) where {T}
    i, n = _pair_from_key(key)
    @inbounds x = _local_target(T, X[i], offsets[n])
    @inbounds el, ori = els[n], oris[n]
    return x, el, ori, n
end

# Data at the reference point `ŷ`: kernel `M = K(x, y(ŷ))`, Lagrange basis `L(ŷ)`, and integration
# measure `τ′(ŷ)`.
@inline function _node_data(K, L, x, el, ori, ŷ)
    y = el(ŷ)
    jac = Inti.jacobian(el, ŷ)
    ν = Inti._normal(jac, ori)
    τ′ = _integration_measure(jac)
    M = K(x, (coords = y, normal = ν))
    return M, L(ŷ), τ′
end

# Value of the integrand given the data at a node; its `k`-th component is `(M * Lv[k]) * τ′`.
@inline _integrand_value(M, Lv, τ′) = map(v -> M * v, Lv) * τ′

# Integrand for the pair encoded in `key`: `ŷ ↦ K(x, y(ŷ)) L(ŷ) τ′(ŷ)` on the reference element.
# Also returns the target and element index.
@inline function _pair_integrand(key, X, els, offsets, oris, K, L, ::Type{T}) where {T}
    x, el, ori, n = _pair_data(key, X, els, offsets, oris, T)
    integrand = ŷ -> _integrand_value(_node_data(K, L, x, el, ori, ŷ)...)
    return integrand, x, n
end

# Write the correction of pair `p` (integrals `W` minus the regular quadrature) in the CSR
# arrays, and the number of subdivisions `nb`, or `-nb - 1` (always negative) if the
# tolerance was not met.
@inline function _write_pair!(nzval, colval, nsub, p, W, nb, converged, x, n, K, ynodes, wnodes, el2qtags)
    np = length(W)
    @inbounds for k in 1:np
        idx = np * (p - 1) + k
        nzval[idx] = W[k] - K(x, ynodes[k, n]) * wnodes[k, n]
        colval[idx] = el2qtags[k, n]
    end
    @inbounds nsub[p] = converged ? Int32(nb) : -Int32(nb) - Int32(1)
    return
end

# The fields of the `EmbeddedRule` are passed to the kernels as separate arguments, and the rule
# is rebuilt inside the kernel: passing the rule itself as a single argument made the CPU
# (POCL) backend crash on x86 (Float32, triangles), while passing its fields does not.
_rule_fields(rule::EmbeddedRule) = (rule.nodes, rule.weights_high, rule.weights_low, rule.domain)

# First tier: panels stored in per-thread arrays of capacity `CAP` (see `_tiers`).
@kernel function _correction_private_kernel!(
        nzval, colval, nsub, @Const(list), @Const(keys), @Const(X), @Const(els),
        @Const(offsets), @Const(oris), @Const(ynodes), @Const(wnodes), @Const(el2qtags),
        K, L, rnodes, rwh, rwl, rdomain, rtol, atol, maxsubdiv,
    )
    t = @index(Global, Linear)
    rule = EmbeddedRule(rnodes, rwh, rwl, rdomain)
    @inbounds p = isnothing(list) ? t : Int(list[t])
    @inbounds f, x, n = _pair_integrand(keys[p], X, els, offsets, oris, K, L, typeof(rtol))
    W, nb, converged = _adaptive_integrate(f, rule, rtol, atol, maxsubdiv)
    _write_pair!(nzval, colval, nsub, p, W, nb, converged, x, n, K, ynodes, wnodes, el2qtags)
end

# Later tiers: panels stored in global memory, one column of the scratch arrays per work-item;
# processes the pairs `list[offset + 1:offset + ndrange]`.
@kernel function _correction_scratch_kernel!(
        nzval, colval, nsub, @Const(list), offset, spanels, sEs, sIs, @Const(keys), @Const(X),
        @Const(els), @Const(offsets), @Const(oris), @Const(ynodes), @Const(wnodes),
        @Const(el2qtags), K, L, rnodes, rwh, rwl, rdomain, rtol, atol, maxsubdiv, store,
    )
    t = @index(Global, Linear)
    rule = EmbeddedRule(rnodes, rwh, rwl, rdomain)
    @inbounds p = Int(list[offset + t])
    @inbounds f, x, n = _pair_integrand(keys[p], X, els, offsets, oris, K, L, typeof(rtol))
    I, E = _apply_rule(f, rule, rule.domain)
    W, nb, converged = _refine!(
        view(spanels, :, t), view(sEs, :, t), view(sIs, :, t), f, rule, I, E, rtol, atol,
        maxsubdiv, store,
    )
    _write_pair!(nzval, colval, nsub, p, W, nb, converged, x, n, K, ynodes, wnodes, el2qtags)
end

## Cooperative integration: one workgroup of `G` work-items per pair
#
# For integrands with many components (e.g. Stokes kernels in 3D), accumulating all of them in
# a single work-item exceeds the available registers. Instead, a workgroup integrates each pair:
# the data at the nodes of the rule (kernel, Lagrange basis, integration measure) is computed in
# parallel and stored in shared memory, and the work-item `t` accumulates the scalar components
# `c = t, t + G, …` of the integral with the same operations, in the same order, as
# `_apply_rule`. The error estimates and norms are computed as by `norm` (the squared norms of
# the blocks, one per Lagrange basis function, in parallel, and their sum by the first work-item),
# and shared with the others. The arithmetic is therefore the same as in
# `_adaptive_integrate`/`_refine!` (without stored panel integrals): the results are identical
# (e.g. on Metal), up to the contraction of operations into fused multiply-adds, which backend
# compilers may perform differently in the two kernels (e.g. the CPU backend). All the
# work-items take the same branches, as required by the barriers.

# scalar entries of a block (value of the kernel times a Lagrange basis function)
_nentries(::Type{V}) where {V <: Number} = 1
_nentries(::Type{V}) where {V <: StaticArray} = length(V)
@inline _entry(v::Number, e) = v
@inline _entry(v::StaticArray, e) = @inbounds v[e]
@inline _from_entries(::Type{V}, f) where {V <: Number} = f(1)
@inline _from_entries(::Type{V}, f) where {V <: StaticArray} = V(ntuple(f, Val(length(V))))

# block `k` of a vector of blocks of type `V` whose scalar components are stored in `s`
@inline _block(::Type{V}, s, k) where {V} = _from_entries(V, e -> @inbounds(s[(k - 1) * _nentries(V) + e]))
@inline _blocks(::Type{V}, s, ::Val{NP}) where {V, NP} = SVector{NP}(ntuple(k -> _block(V, s, k), Val(NP)))

# `norm` of the vector of `NP` blocks of type `V` whose scalar components are stored in `s`
# (shared memory), returned to all the work-items. Same result as `norm` on a `SVector` of
# blocks: the square root of the sum of the squared norms of the blocks, in order, unless it is
# zero or not finite (in which case `norm` itself is used, for its rescaling).
@inline function _coop_norm_of(s, t, ::Val{G}, ::Val{NP}, ::Type{V}, sB, sS, slot) where {G, NP, V}
    @inbounds for k in t:G:NP
        sB[k] = LinearAlgebra.norm_sqr(_block(V, s, k))
    end
    KA.@synchronize()
    if t == 1
        @inbounds begin
            acc = sB[1]
            for k in 2:NP
                acc += sB[k]
            end
            l = sqrt(acc)
            if !(zero(l) < l && isfinite(l))
                l = norm(_blocks(V, s, Val(NP)))
            end
            sS[slot] = l
        end
    end
    KA.@synchronize()
    return @inbounds sS[slot]
end

# Integral over `dom` (scalar components owned by the work-item `t`, stored in `Iown`), and
# error estimate (returned to all the work-items). Same as `_apply_rule`.
@inline function _coop_apply!(Iown, node, rule::EmbeddedRule{H, Lo}, dom, t, g::Val{G}, np::Val{NP}, ::Type{V}, sM, sL, sτ, sC, sB, sS) where {H, Lo, G, NP, V}
    Φ, μ = HAdaptiveIntegration.map_from_reference(dom)
    nodes, wh, wl = rule.nodes, rule.weights_high, rule.weights_low
    NE = _nentries(V)
    # node data, stored as scalars (each work-item then reads the entries it needs)
    @inbounds for i in t:G:H
        M, Lv, τ′ = node(Φ(nodes[i]))
        for e in 1:NE
            sM[e, i] = _entry(M, e)
        end
        for k in 1:NP
            sL[k, i] = Lv[k]
        end
        sτ[i] = τ′
    end
    KA.@synchronize()
    @inbounds for (j, c) in enumerate(t:G:(NP * NE))
        k, e = divrem(c - 1, NE) .+ 1
        v = (sM[e, 1] * sL[k, 1]) * sτ[1]
        Il = wl[1] * v
        Ih = wh[1] * v
        for i in 2:Lo
            v = (sM[e, i] * sL[k, i]) * sτ[i]
            Il += wl[i] * v
            Ih += wh[i] * v
        end
        for i in (Lo + 1):H
            Ih += wh[i] * ((sM[e, i] * sL[k, i]) * sτ[i])
        end
        Iown[j] = μ * Ih
        sC[c] = Ih - Il
    end
    KA.@synchronize()
    return μ * _coop_norm_of(sC, t, g, np, V, sB, sS, 1)
end

# `norm` of the integral whose scalar components are distributed among the work-items
@inline function _coop_norm(Iown, t, g::Val{G}, np::Val{NP}, ::Type{V}, sC, sB, sS) where {G, NP, V}
    @inbounds for (j, c) in enumerate(t:G:(NP * _nentries(V)))
        sC[c] = Iown[j]
    end
    KA.@synchronize()
    return _coop_norm_of(sC, t, g, np, V, sB, sS, 2)
end

# Same as `_refine!` (without stored panel integrals), with the panels and their errors stored in
# `panels` and `Es` (shared or global memory), written by the first work-item.
@inline function _coop_refine!(panels, Es, Iown, node, rule::EmbeddedRule, rtol, atol, maxsubdiv, t, g::Val, np::Val, ::Type{V}, sM, sL, sτ, sC, sB, sS) where {V}
    dom₀ = rule.domain
    NC = _nchildren(typeof(dom₀))
    subdivide = HAdaptiveIntegration.default_subdivision(dom₀)
    shared = (sM, sL, sτ, sC, sB, sS)
    E = _coop_apply!(Iown, node, rule, dom₀, t, g, np, V, shared...)
    if t == 1
        @inbounds panels[1], Es[1] = dom₀, E
    end
    tmp = similar(Iown)
    npanels = 1
    nsubdiv = 0
    converged = (E ≤ atol) || (E ≤ rtol * _coop_norm(Iown, t, g, np, V, sC, sB, sS))
    @inbounds while !converged && nsubdiv < maxsubdiv
        KA.@synchronize() # the panels and errors written by the first work-item are visible
        # find the panel with the largest error
        k = 1
        Emax = Es[1]
        for m in 2:npanels
            if Es[m] > Emax
                k, Emax = m, Es[m]
            end
        end
        dom = panels[k]
        _coop_apply!(tmp, node, rule, dom, t, g, np, V, shared...) # integral over `dom`
        for j in eachindex(Iown)
            Iown[j] -= tmp[j]
        end
        E -= Emax
        children = subdivide(dom)
        for c in 1:NC
            Ec = _coop_apply!(tmp, node, rule, children[c], t, g, np, V, shared...)
            for j in eachindex(Iown)
                Iown[j] += tmp[j]
            end
            E += Ec
            # the first child replaces its parent, the others are appended
            if t == 1
                slot = c == 1 ? k : npanels + c - 1
                panels[slot], Es[slot] = children[c], Ec
            end
        end
        npanels += NC - 1
        nsubdiv += 1
        converged = (E ≤ atol) || (E ≤ rtol * _coop_norm(Iown, t, g, np, V, sC, sB, sS))
    end
    return nsubdiv, converged
end

# Cooperative counterpart of `_write_pair!`: the blocks are gathered in shared memory, and the
# work-item `t` writes the blocks `k = t, t + G, …`.
@inline function _coop_write_pair!(nzval, colval, nsub, p, Iown, nb, converged, x, n, K, ynodes, wnodes, el2qtags, t, ::Val{G}, ::Val{NP}, ::Type{V}, sC) where {G, NP, V}
    @inbounds for (j, c) in enumerate(t:G:(NP * _nentries(V)))
        sC[c] = Iown[j]
    end
    KA.@synchronize()
    @inbounds for k in t:G:NP
        idx = NP * (p - 1) + k
        nzval[idx] = _block(V, sC, k) - K(x, ynodes[k, n]) * wnodes[k, n]
        colval[idx] = el2qtags[k, n]
    end
    if t == 1
        @inbounds nsub[p] = converged ? Int32(nb) : -Int32(nb) - Int32(1)
    end
    return
end

# shared memory of the cooperative kernels: data at the nodes, scalar components, squared norms
# of the blocks, and scalars
macro _coop_shared(VT, T, H, NP, NCOMP)
    return esc(
        quote
            sM = KA.@localmem $T (_nentries($VT), $H)
            sL = KA.@localmem $T ($NP, $H)
            sτ = KA.@localmem $T ($H,)
            sC = KA.@localmem $T ($NCOMP,)
            sB = KA.@localmem $T ($NP,)
            sS = KA.@localmem $T (2,)
        end
    )
end

# First tier: panels stored in shared memory (`CAPP` panels).
@kernel function _correction_coop_kernel!(
        nzval, colval, nsub, @Const(list), @Const(keys), @Const(X), @Const(els),
        @Const(offsets), @Const(oris), @Const(ynodes), @Const(wnodes), @Const(el2qtags),
        K, L, rnodes, rwh, rwl, rdomain, rtol::T, atol, maxsubdiv,
        g::Val{G}, ::Val{CAPP}, ::Val{VT}, ::Val{H}, np::Val{NP}, ::Val{NCOMP},
    ) where {T, G, CAPP, VT, H, NP, NCOMP}
    # (the shared memory is declared with type parameters: `@localmem` declarations are hoisted)
    gid = @index(Group, Linear)
    t = @index(Local, Linear)
    rule = EmbeddedRule(rnodes, rwh, rwl, rdomain)
    @_coop_shared VT T H NP NCOMP
    panels = KA.@localmem typeof(rdomain) (CAPP,)
    Es = KA.@localmem T (CAPP,)
    @inbounds p = isnothing(list) ? gid : Int(list[gid])
    @inbounds x, el, ori, n = _pair_data(keys[p], X, els, offsets, oris, T)
    node = ŷ -> _node_data(K, L, x, el, ori, ŷ)
    Iown = MVector{cld(NCOMP, G), T}(undef)
    nb, converged = _coop_refine!(panels, Es, Iown, node, rule, rtol, atol, maxsubdiv, t, g, np, VT, sM, sL, sτ, sC, sB, sS)
    _coop_write_pair!(nzval, colval, nsub, p, Iown, nb, converged, x, n, K, ynodes, wnodes, el2qtags, t, g, np, VT, sC)
end

# Later tiers: panels stored in global memory, one column of the scratch arrays per workgroup;
# processes the pairs `list[offset + 1:offset + ngroups]`.
@kernel function _correction_coop_scratch_kernel!(
        nzval, colval, nsub, @Const(list), offset, spanels, sEs, @Const(keys), @Const(X),
        @Const(els), @Const(offsets), @Const(oris), @Const(ynodes), @Const(wnodes),
        @Const(el2qtags), K, L, rnodes, rwh, rwl, rdomain, rtol::T, atol, maxsubdiv,
        g::Val{G}, ::Val{VT}, ::Val{H}, np::Val{NP}, ::Val{NCOMP},
    ) where {T, G, VT, H, NP, NCOMP}
    gid = @index(Group, Linear)
    t = @index(Local, Linear)
    rule = EmbeddedRule(rnodes, rwh, rwl, rdomain)
    @_coop_shared VT T H NP NCOMP
    @inbounds p = Int(list[offset + gid])
    @inbounds x, el, ori, n = _pair_data(keys[p], X, els, offsets, oris, T)
    node = ŷ -> _node_data(K, L, x, el, ori, ŷ)
    Iown = MVector{cld(NCOMP, G), T}(undef)
    nb, converged = _coop_refine!(
        view(spanels, :, gid), view(sEs, :, gid), Iown, node, rule, rtol, atol, maxsubdiv, t, g, np,
        VT, sM, sL, sτ, sC, sB, sS,
    )
    _coop_write_pair!(nzval, colval, nsub, p, Iown, nb, converged, x, n, K, ynodes, wnodes, el2qtags, t, g, np, VT, sC)
end

# per-thread memory budget (bytes) for the panels of the first tier
const PRIVATE_BYTES = 4096
# global memory budget (bytes) for the panels of the later tiers
const SCRATCH_BYTES = 2^30

"""
    _tiers(plan, IT; private_bytes = PRIVATE_BYTES)

Capacities (maximum numbers of subdivisions) of the successive tiers of the adaptive
integration, for integrals of type `IT`. All the pairs are first integrated with per-thread
storage for `caps[1]` subdivisions, chosen so that it fits in `private_bytes`. The pairs that
did not converge are then integrated again from scratch with storage in global memory, with
capacities increasing by a factor 8 up to `maxsubdiv`. Since the algorithm is deterministic,
the result for a pair does not depend on the tier in which it converges.
"""
function _tiers(plan::AdaptiveCorrectionPlan, IT; private_bytes = PRIVATE_BYTES)
    T = _float_type(plan)
    DOM = typeof(plan.rule.domain)
    pbytes = _panel_bytes(DOM, T, IT)
    maxsubdiv = _val(plan.maxsubdiv)
    cap = 4
    while 2cap ≤ maxsubdiv && _capacity(DOM, 2cap) * pbytes ≤ private_bytes
        cap *= 2
    end
    caps = [min(cap, maxsubdiv)]
    while caps[end] < maxsubdiv
        push!(caps, min(8 * caps[end], maxsubdiv))
    end
    return caps
end

_panel_bytes(::Type{DOM}, ::Type{T}, ::Type{IT}) where {DOM, T, IT} =
    sizeof(DOM) + sizeof(T) + (_store_integrals(IT, T) ? sizeof(IT) : 0)

@kernel function _failed_flags_kernel!(flags, @Const(list), @Const(nsub))
    t = @index(Global, Linear)
    @inbounds p = isnothing(list) ? t : Int(list[t])
    @inbounds flags[t] = nsub[p] < 0 ? Int32(1) : Int32(0)
end

@kernel function _compact_kernel!(out, @Const(list), @Const(pos), @Const(flags))
    t = @index(Global, Linear)
    @inbounds if flags[t] == 1
        out[pos[t]] = isnothing(list) ? Int32(t) : list[t]
    end
end

# Indices of the pairs in `list` (all `n` pairs if `list === nothing`) that did not converge,
# in the same order as in `list` (so that the ordering by cost of the first tier is preserved
# in the later tiers), and their number.
function _failed_pairs(backend, list, nsub, n)
    flags = KA.allocate(backend, Int32, n)
    _failed_flags_kernel!(backend)(flags, list, nsub; ndrange = n)
    pos = similar(flags)
    AK.accumulate!(+, pos, flags; init = Int32(0), inclusive = true)
    nfail = Int(Array(view(pos, n:n))[1])
    out = KA.allocate(backend, Int32, max(nfail, 1))
    nfail > 0 && _compact_kernel!(backend)(out, list, pos, flags; ndrange = n)
    return out, nfail
end

"""
    _nearfield_values(plan, X, keys, npairs; sort_by_cost, private_bytes, scratch_bytes, cooperative, group_size)

Compute the values (and column indices) of the CSR correction matrix, together with the
number of subdivisions used for each pair. The adaptive integration is performed in tiers of
increasing capacity, see [`_tiers`](@ref).

If `cooperative`, each pair is integrated by a workgroup of `group_size` work-items (see
`_coop_refine!`) instead of a single work-item, with the same arithmetic. By default
(`cooperative = :auto`), this is done on GPUs for integrands with many components.

If `sort_by_cost`, the pairs of the first tier are processed in the order given by
[`_cost_order`](@ref). This is the default on GPUs, where it reduces the divergence between
work-items executed in lockstep; on the CPU backend the original order (pairs of the same
target and neighboring elements are adjacent) is more cache-friendly.
"""
function _nearfield_values(
        plan::AdaptiveCorrectionPlan, X, keys, npairs;
        sort_by_cost = !(plan.backend isa typeof(KA.CPU())),
        private_bytes = PRIVATE_BYTES, scratch_bytes = SCRATCH_BYTES,
        cooperative = :auto, group_size = 32,
    )
    backend = plan.backend
    np = size(plan.el2qtags, 1)
    T = _float_type(plan)
    XT = Base.promote_op(_local_target, Type{T}, eltype(X), eltype(plan.offsets))
    VT = Inti.return_type(plan.kernel, XT, eltype(plan.ynodes))
    IT = SVector{np, VT}
    nzval = KA.allocate(backend, VT, np * npairs)
    colval = KA.allocate(backend, Int32, np * npairs)
    nsub = KA.allocate(backend, Int32, npairs)
    npairs == 0 && return colval, nzval, nsub
    args = (
        keys, X, plan.els, plan.offsets, plan.oris, plan.ynodes, plan.wnodes, plan.el2qtags,
        plan.kernel, plan.basis, _rule_fields(plan.rule)..., plan.rtol, plan.atol,
    )
    caps = _tiers(plan, IT; private_bytes)
    coop = cooperative === :auto ? _cooperative_default(backend, IT, T) : cooperative
    DOM = typeof(plan.rule.domain)
    list = sort_by_cost ? _cost_order(plan, X, keys, npairs) : nothing
    if coop
        G = group_size
        cvals = (Val(G), Val(VT), Val(length(plan.rule.weights_high)), Val(np), Val(np * _nentries(VT)))
        # first tier: all pairs, panels in shared memory
        _correction_coop_kernel!(backend, G)(
            nzval, colval, nsub, list, args..., caps[1], cvals[1], Val(_capacity(DOM, caps[1])),
            cvals[2:end]...; ndrange = npairs * G,
        )
        # later tiers: pairs that did not converge, panels in global memory
        list, nlist = _failed_pairs(backend, list, nsub, npairs)
        for cap in caps[2:end]
            nlist == 0 && break
            ncap = _capacity(DOM, cap)
            batch = clamp(scratch_bytes ÷ (ncap * (sizeof(DOM) + sizeof(T))), 1, nlist)
            spanels = KA.allocate(backend, DOM, ncap, batch)
            sEs = KA.allocate(backend, T, ncap, batch)
            for offset in 0:batch:(nlist - 1)
                nb = min(batch, nlist - offset)
                _correction_coop_scratch_kernel!(backend, G)(
                    nzval, colval, nsub, list, offset, spanels, sEs, args..., cap, cvals...;
                    ndrange = nb * G,
                )
            end
            list, nlist = _failed_pairs(backend, list, nsub, nlist)
        end
        return colval, nzval, nsub
    end
    # first tier: all pairs, per-thread storage
    _correction_private_kernel!(backend)(nzval, colval, nsub, list, args..., Val(caps[1]); ndrange = npairs)
    # later tiers: pairs that did not converge, storage in global memory
    list, nlist = _failed_pairs(backend, list, nsub, npairs)
    store = Val(_store_integrals(IT, T))
    for cap in caps[2:end]
        nlist == 0 && break
        ncap = _capacity(DOM, cap)
        batch = clamp(scratch_bytes ÷ (ncap * _panel_bytes(DOM, T, IT)), 1, nlist)
        spanels = KA.allocate(backend, DOM, ncap, batch)
        sEs = KA.allocate(backend, T, ncap, batch)
        # (one unused row if the integrals are not stored, to avoid empty allocations)
        sIs = KA.allocate(backend, IT, _store_integrals(IT, T) ? ncap : 1, batch)
        for offset in 0:batch:(nlist - 1)
            nb = min(batch, nlist - offset)
            _correction_scratch_kernel!(backend)(
                nzval, colval, nsub, list, offset, spanels, sEs, sIs, args..., cap, store;
                ndrange = nb,
            )
        end
        list, nlist = _failed_pairs(backend, list, nsub, nlist)
    end
    return colval, nzval, nsub
end

# Whether to use the cooperative integration by default: on GPUs, for integrands with many
# components (the same criterion as for storing the integrals over the panels). On Metal, for
# Stokes kernels in 3D, it is about 3× faster than the per-thread integration, and slightly
# slower for the Laplace kernels. The CPU backend executes the work-items of a workgroup
# sequentially, so that the per-thread integration is preferable there.
function _cooperative_default(backend, ::Type{IT}, ::Type{T}) where {IT, T}
    return !(backend isa typeof(KA.CPU())) && !_store_integrals(IT, T)
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
