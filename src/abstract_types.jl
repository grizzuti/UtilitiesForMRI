# Abstract types


## Spatial geometry

abstract type AbstractSpatialGeometry{T<:Real} end
abstract type AbstractCartesianSpatialGeometry{T<:Real}<:AbstractSpatialGeometry{T} end


## k-space acquisition geometry

abstract type AbstractKSpaceSampling{T<:Real} end
abstract type AbstractStructuredKSpaceSampling{T<:Real}<:AbstractKSpaceSampling{T} end


## Fourier operators

abstract type AbstractFourierTransform{T<:Real, XT<:AbstractSpatialGeometry{T}, KT<:AbstractKSpaceSampling{T}, N, M}<:AbstractLinearOperator{Complex{T}, N, M} end

abstract type AbstractRigidMotionPerturbationFourierTransform{T<:Real, XT<:AbstractSpatialGeometry{T}, KT<:AbstractKSpaceSampling{T}, N, M}<:AbstractLinearOperator{Complex{T}, N, M} end