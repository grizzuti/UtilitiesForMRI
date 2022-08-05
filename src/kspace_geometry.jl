# k-space geometry utilities

export KSpaceSampling, StructuredKSpaceSampling, CartesianStructuredKSpaceSampling, kspace_sampling, coord, destructure


## k-space sampling (= ordered wave-number coordinates)

struct KSpaceSampling{T}<:AbstractKSpaceSampling{T}
    coord::AbstractArray{T,2} # k-space coordinates size=(nt,3)
end

kspace_sampling(coord::AbstractArray{T,2}) where {T<:Real} = KSpaceSampling{T}(coord)

coord(K::KSpaceSampling{T}) where {T<:Real} = K.coord


## Structured k-space sampling

struct StructuredKSpaceSampling{T}<:AbstractStructuredKSpaceSampling{T}
    permutation_dims::NTuple{3,Integer}
    coord_phase_encoding::AbstractArray{T,2} # k-space coordinates size=(nt,2)
    coord_readout::AbstractVector{T} # k-space coordinates size=(nk,)
end

kspace_sampling(permutation_dims::NTuple{3,Integer}, coord_phase_encoding::AbstractArray{T,2}, coord_readout::AbstractVector{T}) where {T<:Real} = StructuredKSpaceSampling{T}(permutation_dims, coord_phase_encoding, coord_readout)

coord_phase_encoding(K::StructuredKSpaceSampling) = K.coord_phase_encoding
coord_readout(K::StructuredKSpaceSampling) = K.coord_readout

function coord(K::AbstractStructuredKSpaceSampling{T}) where {T<:Real}
    nt, nk = size(K)
    perm = dims_permutation(K)
    k_pe, k_r = coord_phase_encoding(K), coord_readout(K)
    k_coordinates = cat(repeat(reshape(k_pe, nt,1,2); outer=(1,nk,1)), repeat(reshape(k_r, 1,nk,1); outer=(nt,1,1)); dims=3)[:,:,invperm(perm)]
    return k_coordinates
end

Base.size(K::AbstractStructuredKSpaceSampling) = (size(coord_phase_encoding(K),1),length(coord_readout(K)))

function Base.getindex(K::AbstractStructuredKSpaceSampling{T}, t::Integer) where {T<:Real}
    _, nk = size(K)
    return cat(repeat(reshape(coord_phase_encoding(K)[t,:],1,2); outer=(nk,1)), reshape(coord_readout(K),nk,1); dims=2)[:,invperm(dims_permutation(K))]
end

destructure(K::AbstractStructuredKSpaceSampling{T}) where {T<:Real}= kspace_sampling(reshape(coord(K),:,3))


## Cartesian structured k-space sampling

struct CartesianStructuredKSpaceSampling{T}<:AbstractStructuredKSpaceSampling{T}
    spatial_geometry::CartesianSpatialGeometry{T}
    permutation_dims::NTuple{3,Integer}
    idx_phase_encoding::AbstractVector{<:Integer} # k-space index coordinates size=(nt,2)
    idx_readout::AbstractVector{<:Integer} # k-space index coordinates size=(nk,)
end

function kspace_sampling(X::CartesianSpatialGeometry{T}, phase_encoding_dims::NTuple{2,Integer}; phase_encode_sampling::Union{Nothing,AbstractVector{<:Integer}}=nothing, readout_sampling::Union{Nothing,AbstractVector{<:Integer}}=nothing) where {T<:Real}

    permutation_dims = dims_permutation(phase_encoding_dims)
    n_pe1, n_pe2, n_r = size(X)[permutation_dims]
    isnothing(phase_encode_sampling) && (phase_encode_sampling = 1:n_pe1*n_pe2)
    isnothing(readout_sampling)      && (readout_sampling = 1:n_r)
    return CartesianStructuredKSpaceSampling{T}(X, tuple(permutation_dims...), phase_encode_sampling, readout_sampling)

end

function coord(K::CartesianStructuredKSpaceSampling{T}) where {T<:Real}
    nt, nk = size(K)
    perm = dims_permutation(K)
    k_pe, k_r = coord_phase_encoding(K), coord_readout(K)
    k_coordinates = cat(repeat(reshape(k_pe, nt,1,2); outer=(1,nk,1)), repeat(reshape(k_r, 1,nk,1); outer=(nt,1,1)); dims=3)[:,:,invperm(perm)]
    return k_coordinates
end

function coord_phase_encoding(K::CartesianStructuredKSpaceSampling)
    k_pe1, k_pe2 = k_coord(K.spatial_geometry; mesh=false)[dims_permutation(K)][1:2]
    n_pe1 = length(k_pe1); n_pe2 = length(k_pe2)
    return reshape(cat(repeat(reshape(k_pe1,:,1); outer=(1,n_pe2)), repeat(reshape(k_pe2,1,:); outer=(n_pe1,1)); dims=3),:,2)[K.idx_phase_encoding,:]
end

function coord_readout(K::CartesianStructuredKSpaceSampling)
    k_r = k_coord(K.spatial_geometry; mesh=false)[dims_permutation(K)][3]
    return k_r[K.idx_readout]
end


## Utils

dims_permutation(K::AbstractStructuredKSpaceSampling) = [K.permutation_dims...]

dims_permutation(phase_encoding_dims::NTuple{2,Integer}) = [phase_encoding_dims..., readout_dim(phase_encoding_dims)]

function readout_dim(phase_encoding::NTuple{2,Integer})
    ((phase_encoding == (1,2)) || (phase_encoding == (2,1))) && (readout = 3)
    ((phase_encoding == (1,3)) || (phase_encoding == (3,1))) && (readout = 2)
    ((phase_encoding == (2,3)) || (phase_encoding == (3,2))) && (readout = 1)
    return readout
end