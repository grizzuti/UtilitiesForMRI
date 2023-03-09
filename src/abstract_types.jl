# Abstract types


## Spatial geometry

abstract type AbstractSpatialGeometry{T<:Real} end
abstract type AbstractCartesianSpatialGeometry{T<:Real}<:AbstractSpatialGeometry{T} end


## k-space acquisition geometry

abstract type AbstractKSpaceSampling{T<:Real} end

abstract type AbstractKSpaceSampling3D{T<:Real}<:AbstractKSpaceSampling{T} end # 3D
abstract type AbstractStructuredKSpaceSampling3D{T<:Real}<:AbstractKSpaceSampling3D{T} end

abstract type AbstractKSpaceSampling2DMS{T<:Real}<:AbstractKSpaceSampling{T} end # 2D-multislice
abstract type AbstractStructuredKSpaceSampling2DMS{T<:Real}<:AbstractKSpaceSampling2DMS{T} end


## Fourier operators

abstract type AbstractFourierTransform{T<:Real, XT<:AbstractSpatialGeometry{T}, KT<:AbstractKSpaceSampling{T}, N, M}<:AbstractLinearOperator{Complex{T}, N, M} end

abstract type AbstractRigidMotionPerturbationFourierTransform{T<:Real, XT<:AbstractSpatialGeometry{T}, KT<:AbstractKSpaceSampling{T}, N, M}<:AbstractLinearOperator{Complex{T}, N, M} end