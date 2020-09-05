# SPDX-License-Identifier: MIT
# Copyright (c) 2020: Pablo Zubieta

module ReactionCoordinates


using LinearAlgebra
using StaticArrays


export Acylindricity, Angle, Asphericity, Barycenter, DihedralAngle, DistanceFrom,
    GyrationTensor, PairwiseKernel, PrincipalMoments, RadiusOfGyration,
    RouseMode, Separation, ShapeAnysotropy, TorsionAngle, WeightedBarycenter,
    WeightedGyrationTensor, WeightedRadiusOfGyration


### Type definitions

abstract type ReactionCoordinate <: Function end
abstract type AbstractGyrationTensor <: ReactionCoordinate end
abstract type AbstractBarycenter <: ReactionCoordinate end

struct Angle <: ReactionCoordinate end
struct DihedralAngle <: ReactionCoordinate end
struct GyrationTensor <: AbstractGyrationTensor end
struct RadiusOfGyration{T} <: ReactionCoordinate end
struct Separation{T} <: ReactionCoordinate end

const TorsionAngle = DihedralAngle

struct WeightedGyrationTensor{T} <: AbstractGyrationTensor
    ws::Vector{T}
end

struct WeightedRadiusOfGyration{T} <: ReactionCoordinate
    ws::Vector{T}
end

struct PrincipalMoments{T <: AbstractGyrationTensor} <: ReactionCoordinate
    S::T
end

struct Asphericity{T <: AbstractGyrationTensor} <: ReactionCoordinate
    S::T
end

struct Acylindricity{T <: AbstractGyrationTensor} <: ReactionCoordinate
    S::T
end

struct ShapeAnysotropy{T <: AbstractGyrationTensor} <: ReactionCoordinate
    S::T
end

struct Barycenter{F} <: AbstractBarycenter
    f::F
end

struct WeightedBarycenter{F, T} <: AbstractBarycenter
    f::F
    ws::Vector{T}
end

struct PairwiseKernel{F} <: ReactionCoordinate
    f::F
end

struct DistanceFrom{T} <: ReactionCoordinate
    r::SVector{T}
end

struct RouseMode{N, P, T} <: ReactionCoordinate
    c::T
    ws::Vector{T}

    function RouseMode{N, 0}(::Type{T} = Float64) where {N, T}
        @assert (N > 0) "Parameters must satisfy N > P ≥ 0. Got N = $N and P = 0"
        c = sqrt(one(T) / N)
        return new{N, 0, T}(c, T[])
    end

    function RouseMode{N, P}(::Type{T} = Float64) where {N, P, T}
        @assert (N > P ≥ 0) "Parameters must satisfy N > P ≥ 0. Got N = $N and P = $P"
        c = sqrt(2 * one(T) / N)
        ws = T[cospi(P * (2i - 1) / 2N) for i = 1:N]
        return new{N, P, T}(c, ws)
    end
end

struct TransformMatrix{T} <: FieldMatrix{3, 3, T}
    xx::T
    yx::T
    zx::T
    xy::T
    yy::T
    zy::T
    xz::T
    yz::T
    zz::T
end

#=

Explore using FunctionalWrappers.jl for this

struct MultiReactionCoordinate{T <: Tuple} <: ReactionCoordinate
    ξs::T
end

=#

### Outer Constructors

function WeightedGyrationTensor(ws::Vector)
    w̃s = ws / sum(ws)  # weights normalized at construction
    T = eltype(w̃s)
    return WeightedGyrationTensor{T}(w̃s)
end

function WeightedRadiusOfGyration(ws::Vector)
    w̃s = ws / sum(ws)
    T = eltype(w̃s)
    return WeightedRadiusOfGyration{T}(w̃s)
end

function WeightedBarycenter(ws::Vector; f::F = identity) where {F}
    w̃s = ws / sum(ws)
    T = eltype(w̃s)
    return WeightedBarycenter{F, T}(f, w̃s)
end

PrincipalMoments() = PrincipalMoments(GyrationTensor())
PrincipalMoments(ws::Vector) = PrincipalMoments(WeightedGyrationTensor(ws))

Asphericity() = Asphericity(GyrationTensor())
Asphericity(ws::Vector) = Asphericity(WeightedGyrationTensor(ws))

Acylindricity() = Acylindricity(GyrationTensor())
Acylindricity(ws::Vector) = Acylindricity(WeightedGyrationTensor(ws))

ShapeAnysotropy() = ShapeAnysotropy(GyrationTensor())
ShapeAnysotropy(ws::Vector) = ShapeAnysotropy(WeightedGyrationTensor(ws))

Barycenter() = Barycenter(identity)


### Getters

gyration_tensor(ξ::PrincipalMoments) = ξ.S
gyration_tensor(ξ::Asphericity) = ξ.S
gyration_tensor(ξ::Acylindricity) = ξ.S
gyration_tensor(ξ::ShapeAnysotropy) = ξ.S

weights(ξ::WeightedGyrationTensor) = ξ.ws
weights(ξ::WeightedRadiusOfGyration) = ξ.ws
weights(ξ::WeightedBarycenter) = ξ.ws
weights(ξ::RouseMode) = ξ.ws

op(ξ::AbstractBarycenter) = ξ.f
op(ξ::PairwiseKernel) = ξ.f

reference(ξ::DistanceFrom) = ξ.r

coeff(ξ::RouseMode) = ξ.c

### Functors methods

(ξ::Angle)(rs) = @inbounds angle(rs[1], rs[2], rs[3])
(ξ::DihedralAngle)(rs) = @inbounds dihedral_angle(rs[1], rs[2], rs[3], rs[4])
(ξ::GyrationTensor)(rs) = gyration_tensor(rs)
(ξ::WeightedGyrationTensor)(rs) = gyration_tensor(rs, weights(ξ))
(ξ::RadiusOfGyration)(rs) = radius_of_gyration(rs)
(ξ::WeightedRadiusOfGyration)(rs) = radius_of_gyration(rs, weights(ξ))
(ξ::PrincipalMoments)(rs) = principal_moments(gyration_tensor(ξ)(rs))
(ξ::Asphericity)(rs) = asphericity(gyration_tensor(ξ)(rs))
(ξ::Acylindricity)(rs) = acylindricity(gyration_tensor(ξ)(rs))
(ξ::ShapeAnysotropy)(rs) = shape_anysotropy(gyration_tensor(ξ)(rs))
(ξ::Barycenter)(rs) = barycenter(op(ξ), rs)
(ξ::WeightedBarycenter)(rs) = barycenter(op(ξ), rs, weights(ξ))
(ξ::PairwiseKernel)(rs₁, rs₂) = pairwise(op(ξ), rs₁, rs₂)
(ξ::DistanceFrom)(rs) = distance(rs, reference(ξ))
(ξ::Separation)(rs) = @inbounds distance(rs[1], rs[2])
(ξ::RouseMode)(rs) = rouse_mode(rs, coeff(ξ), weights(ξ))
(ξ::RouseMode{N, 0})(rs) where {N} = rouse_mode₀(rs, coeff(ξ))


#==========#
#  Angles  #
#==========#

"""    angle(r₁, r₂, r₃)

Computes the angle between the two vectors defined by three points in space (around the
point in the middle).
"""
angle(r₁, r₂, r₃) = angle(r₁ - r₂, r₃ - r₂)
#
@inline angle(a, b) = atan(norm(a × b), a ⋅ b)

"""    dihedral_angle(r₁, r₂, r₃, r₄)

Computes the dihedral (or torsion) angle defined by four points in space (around the line
defined by the two central points).
"""
dihedral_angle(r₁, r₂, r₃, r₄) = dihedral_angle(r₂ - r₁, r₃ - r₂, r₄ - r₃)
#
@inline function dihedral_angle(a, b, c)
    p = a × b
    q = b × c
    return atan((p × q) ⋅ b, (p ⋅ q) * norm(b))
end

#==============#
#  Box Volume  #
#==============#

volume(H::TransformMatrix) = det(H)

#=====================#
#  Shape Descriptors  #
#=====================#

function gyration_tensor(rs)
    # Alternative implementation:
    # f = r -> r .* r'
    # S = sum(f, rs)
    S = accumulator(GyrationTensor, rs)
    @simd for r in rs
        S .= muladd.(r, r', S)
    end
    return Symmetric(SMatrix(S ./= length(rs)))
end

function gyration_tensor(rs, ws)
    # Alternative implementation:
    # f = ((w, r),) -> w .* r .* r'
    # S = sum(f, rs)
    S = accumulator(WeightedGyrationTensor, rs, ws)
    @inbounds @simd for i in eachindex(ws, rs)
        w, r = ws[i], rs[i]
        S .= muladd.(w, r .* r', S)
    end
    return Symmetric(SMatrix(S))
end

radius_of_gyration(rs) = sum(r -> r ⋅ r, rs) / length(rs)

function radius_of_gyration(rs, ws)
    # Alternative implementation:
    # f = ((w, r),) -> w * (r ⋅ r)
    # R² = sum(f, rs)
    R² = accumulator(WeightedRadiusOfGyration, rs, ws)
    @inbounds @simd for i in eachindex(ws, rs)
        w, r = ws[i], rs[i]
        R² = muladd(w, r ⋅ r, R²)
    end
    return R²
end

const principal_moments = LinearAlgebra.eigvals

function asphericity(S)
    λ₁², λ₂², λ₃² = principal_moments(S)
    return λ₃² - (λ₁² + λ₂²) / 2
end

function acylindricity(S)
    λ₁², λ₂², λ₃² = principal_moments(S)
    return (λ₂² - λ₁²)
end

function shape_anysotropy(S)
    λ₁², λ₂², λ₃² = principal_moments(S)
    λ₁⁴ = λ₁²^2
    λ₂⁴ = λ₂²^2
    λ₃⁴ = λ₃²^2
    return (3 * (λ₁⁴ + λ₂⁴ + λ₃⁴) / (λ₁² + λ₂² + λ₃²)^2 - 1) / 2
end

### Accumulators

@inline function accumulator(::Type{GyrationTensor}, rs)
    R = eltype(eltype(rs))
    T = typeof(zero(R) / 1)
    return zeros(MMatrix{3, 3, T})
end

@inline function accumulator(::Type{WeightedGyrationTensor}, rs, ws)
    W = eltype(ws)
    R = eltype(eltype(rs))
    T = typeof(zero(W) * zero(R))
    return zeros(MMatrix{3, 3, T})
end

@inline function accumulator(::Type{WeightedRadiusOfGyration}, rs, ws)
    W = eltype(ws)
    R = eltype(eltype(rs))
    T = typeof(zero(W) * zero(R))
    return zero(T)
end

#========================#
#  Particle Coordinates  #
#========================#

const Identity = typeof(identity)

getx(r) = @inbounds(r[1])
gety(r) = @inbounds(r[2])
getz(r) = @inbounds(r[3])

barycenter(f::F, rs) where {F} = sum(f, rs) / length(rs)

function barycenter(::Identity, rs, ws)
    # Alternative implementation:
    # f = ((w, r),) -> w * r
    # R = sum(f, rs)
    R = accumulator(WeightedBarycenter{Identity}, rs, ws)
    @inbounds @simd for i in eachindex(ws, rs)
        w, r = ws[i], rs[i]
        R .= muladd.(w, r, R)
    end
    return SVector(R)
end

function barycenter(f::F, rs, ws) where {F}
    # Alternative implementation:
    # g = ((w, r),) -> w * f(r)
    # R = sum(g, rs)
    R = accumulator(WeightedBarycenter, rs, ws)
    @inbounds @simd for i in eachindex(ws, rs)
        w, r = ws[i], rs[i]
        R = muladd(w, f(r), R)
    end
    return R
end

### Accumulators

@inline function accumulator(::Type{CV}, rs, ws) where {CV <: WeightedBarycenter}
    W = eltype(ws)
    R = eltype(eltype(rs))
    T = typeof(zero(W) * zero(R) / 1)
    return _accumulator(CV, T)
end

@inline _accumulator(::Type{<:WeightedBarycenter}, ::Type{T}) where {T} = zero(T)
@inline function _accumulator(::Type{<:WeightedBarycenter{Identity}}, ::Type{T}) where {T}
    return zeros(MVector{3, T})
end

#====================#
#  Pairwise Kernels  #
#====================#

function pairwise(f::F, r₁, r₂) where {F}
    ξ = accumulator(PairwiseKernel, f, r₁, r₂)
    @inbounds for i in eachindex(r₁)
        @simd for j in eachindex(r₂)
            ξ += f(r₁[i], r₂[j])
        end
    end
    return ξ
end

@inline function accumulator(::Type{PairwiseKernel}, f::F, r₁, r₂) where {F}
    T₁ = eltype(eltype(r₁))
    T₂ = eltype(eltype(r₂))
    T = typeof(f(zero(T₁), zero(T₂)))
    return zero(T)
end

### Common kernels

struct Gaussian{T} <: Function
    μ::T
    σ²::T
end

function Gaussian(μ, σ)
    μ̃, σ̃² = promote(μ, σ^2)
    T = typeof(μ̃)
    return Gaussian{T}(μ̃, σ̃²)
end

mean(f::Gaussian) = f.μ
var(f::Gaussian) = f.σ²

(f::Gaussian)(r) = gaussian(mean(f), var(f), r)

gaussian(μ, σ², r) = exp(-(r - μ)^2 / 2σ²)

#=============#
#  Distances  #
#=============#

distance(r, s) = norm(r - s)

#=============#
#  Utilities  #
#=============#

abstract type DecayingFunction <: Function end

struct RationalDecay{M, N, T} <: DecayingFunction
    d::T
    r::T

    function RationalDecay{M, N}(d, r) where {M, N}
        @assert N > M > 0
        d, r = promote(d₀, r₀)
        return new{M, N, typeof(d)}(d, r)
    end
end

numerator_exponent(::RationalDecay{M}) where {M} = M
denominator_exponent(::RationalDecay{M, N}) where {M, N} = N

origin(f::RationalDecay) = f.d
width(f::RationalDecay) = f.r

RationalDecay(; d₀ = 0.0, r₀ = 1.0) = RationalDecay{6, 12}(d₀, r₀)
RationalDecay{M, N}(; d₀ = 0.0, r₀ = 1.0) where {M, N} = RationalDecay{M, N}(d₀, r₀)

(f::RationalDecay{6, 12})(r) = rational_decay⁶₁₂(r; d₀ = origin(f), r₀ = width(f))
(f::RationalDecay{8, 12})(r) = rational_decay⁸₁₂(r; d₀ = origin(f), r₀ = width(f))
function (f::RationalDecay)(r)
    m = numerator_exponent(f)
    n = denominator_exponent(f)
    return rational_decay(m, n, r; d₀ = origin(f), r₀ = width(f))
end

### More accurate implementation for M = 6, N = 12
@inline function rational_decay⁶₁₂(r; d₀ = 0.0, r₀ = 1.0)
    ρ = ((r - d₀) / r₀)
    if ρ < 0
        return one(ρ)
    end
    return 1 / (1 + ρ^6)
end

### More accurate implementation for M = 8, N = 12
@inline function rational_decay⁸₁₂(r; d₀ = 0.0, r₀ = 1.0)
    ρ = ((r - d₀) / r₀)
    if ρ < 0
        return one(ρ)
    end
    ρ⁴ = ρ^4
    return 1 / (ρ⁴ + 1 / (1 + ρ⁴))
end

@inline function rational_decay(m, n, r; d₀ = 0.0, r₀ = 1.0)
    @assert n > m > 0
    ρ = (r - d₀) / r₀
    if ρ < 0
        return one(ρ)
    elseif isone(ρ)
        return one(ρ) * m / n
    end
    ρᵐ = ρ^m
    if isinf(ρᵐ)
        return zero(ρ)
    end
    return (1 - ρᵐ) / (1 - ρ^n)
end

#===============#
#  Rouse Modes  #
#===============#

function rouse_modeₒ(rs, c)
    x₀ = sum(rs)
    return c * norm(x₀)
end
#
function rouse_mode(rs, c, ws)
    xₚ = sum(((w, r),) -> w * r, zip(ws, rs))
    return c * norm(xₚ)
end


end  # module ReactionCoordinates
