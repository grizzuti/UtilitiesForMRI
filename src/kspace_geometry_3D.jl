# 3D k-space geometry utilities

export KSpaceSampling3D, SubsampledKSpaceSampling3D, StructuredKSpaceSampling3D, SubsampledStructuredKSpaceSampling3D, CartesianStructuredKSpaceSampling3D, SubsampledCartesianStructuredKSpaceSampling3D, kspace_sampling, coord, isa_subsampling, phase_encoding_dims, readout_dim


## k-space sampling (= ordered wave-number coordinates)

struct KSpaceSampling3D{T}<:AbstractKSpaceSampling3D{T}
    coord::AbstractArray{T,2} # k-space coordinates size=(nt,3)
end

kspace_sampling(coord::AbstractArray{T,2}) where {T<:Real} = KSpaceSampling3D{T}(coord)

coord(K::KSpaceSampling3D{T}) where {T<:Real} = K.coord


## Structured k-space sampling

struct StructuredKSpaceSampling3D{T}<:AbstractStructuredKSpaceSampling3D{T}
    permutation_dims::NTuple{3,Integer}
    coord_phase_encoding::AbstractArray{T,2} # k-space coordinates size=(nt,2)
    coord_readout::AbstractVector{T} # k-space coordinates size=(nk,)
end

kspace_sampling(permutation_dims::NTuple{3,Integer}, coord_phase_encoding::AbstractArray{T,2}, coord_readout::AbstractVector{T}) where {T<:Real} = StructuredKSpaceSampling3D{T}(permutation_dims, coord_phase_encoding, coord_readout)

coord_phase_encoding(K::StructuredKSpaceSampling3D) = K.coord_phase_encoding
coord_readout(K::StructuredKSpaceSampling3D) = K.coord_readout

function coord(K::AbstractStructuredKSpaceSampling3D{T}) where {T<:Real}
    nt, nk = size(K)
    perm = dims_permutation(K)
    k_pe, k_r = coord_phase_encoding(K), coord_readout(K)
    k_coordinates = cat(repeat(reshape(k_pe, nt,1,2); outer=(1,nk,1)), repeat(reshape(k_r, 1,nk,1); outer=(nt,1,1)); dims=3)[:,:,invperm(perm)]
    return k_coordinates
end

Base.size(K::AbstractStructuredKSpaceSampling3D) = (size(coord_phase_encoding(K),1),length(coord_readout(K)))

function Base.getindex(K::AbstractStructuredKSpaceSampling3D{T}, t::Integer) where {T<:Real}
    _, nk = size(K)
    return cat(repeat(reshape(coord_phase_encoding(K)[t,:],1,2); outer=(nk,1)), reshape(coord_readout(K),nk,1); dims=2)[:,invperm(dims_permutation(K))]
end

Base.convert(::Type{KSpaceSampling3D{T}}, K::AbstractStructuredKSpaceSampling3D{T}) where {T<:Real} = kspace_sampling(reshape(coord(K),:,3))
Base.convert(::Type{KSpaceSampling3D}, K::AbstractStructuredKSpaceSampling3D{T}) where {T<:Real} = convert(KSpaceSampling{T}, K)

phase_encoding_dims(K::AbstractStructuredKSpaceSampling3D) = K.permutation_dims[1:2]
readout_dim(K::AbstractStructuredKSpaceSampling3D) = K.permutation_dims[3]


## Cartesian structured k-space sampling

struct CartesianStructuredKSpaceSampling3D{T}<:AbstractStructuredKSpaceSampling3D{T}
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
    return CartesianStructuredKSpaceSampling3D{T}(X, tuple(permutation_dims...), phase_encode_sampling, readout_sampling)

end

function coord(K::CartesianStructuredKSpaceSampling3D{T}) where {T<:Real}
    nt, nk = size(K)
    perm = dims_permutation(K)
    k_pe, k_r = coord_phase_encoding(K), coord_readout(K)
    k_coordinates = cat(repeat(reshape(k_pe, nt,1,2); outer=(1,nk,1)), repeat(reshape(k_r, 1,nk,1); outer=(nt,1,1)); dims=3)[:,:,invperm(perm)]
    return k_coordinates
end

function coord_phase_encoding(K::CartesianStructuredKSpaceSampling3D)
    k_pe1, k_pe2 = k_coord(K.spatial_geometry; mesh=false)[dims_permutation(K)][1:2]
    n_pe1 = length(k_pe1); n_pe2 = length(k_pe2)
    return reshape(cat(repeat(reshape(k_pe1,:,1); outer=(1,n_pe2)), repeat(reshape(k_pe2,1,:); outer=(n_pe1,1)); dims=3),:,2)[K.idx_phase_encoding,:]
end

function coord_readout(K::CartesianStructuredKSpaceSampling3D)
    k_r = k_coord(K.spatial_geometry; mesh=false)[dims_permutation(K)][3]
    return k_r[K.idx_readout]
end

Base.convert(::Type{StructuredKSpaceSampling3D{T}}, K::CartesianStructuredKSpaceSampling3D{T}) where {T<:Real} = kspace_sampling(K.permutation_dims, coord_phase_encoding(K), coord_readout(K))
Base.convert(::Type{StructuredKSpaceSampling3D}, K::CartesianStructuredKSpaceSampling3D{T}) where {T<:Real} = convert(StructuredKSpaceSampling{T}, K)


## Utils

dims_permutation(K::AbstractStructuredKSpaceSampling3D) = [K.permutation_dims...]

dims_permutation(phase_encoding_dims::NTuple{2,Integer}) = [phase_encoding_dims..., readout_dim(phase_encoding_dims)]

function readout_dim(phase_encoding::NTuple{2,Integer})
    ((phase_encoding == (1,2)) || (phase_encoding == (2,1))) && (readout = 3)
    ((phase_encoding == (1,3)) || (phase_encoding == (3,1))) && (readout = 2)
    ((phase_encoding == (2,3)) || (phase_encoding == (3,2))) && (readout = 1)
    return readout
end


## Sub-sampled k-space

struct SubsampledKSpaceSampling3D{T<:Real}<:AbstractKSpaceSampling3D{T}
    kspace_sampling::KSpaceSampling3D{T}
    subindex::Union{Colon,AbstractVector{<:Integer}}
end

Base.getindex(K::KSpaceSampling3D{T}, subindex::Union{Colon,AbstractVector{<:Integer}}) where {T<:Real} = SubsampledKSpaceSampling3D{T}(K, subindex)
Base.getindex(K::SubsampledKSpaceSampling3D{T}, subindex::Union{Colon,AbstractVector{<:Integer}}) where {T<:Real} = K.kspace_sampling[K.subindex[subindex]]

isa_subsampling(Kq::SubsampledKSpaceSampling3D{T}, K::KSpaceSampling3D{T}) where {T<:Real} = (Kq.kspace_sampling == K)
isa_subsampling(Kq::SubsampledKSpaceSampling3D{T}, K::SubsampledKSpaceSampling3D{T}) where {T<:Real} = (Kq.kspace_sampling == K.kspace_sampling) && all(Kq.subindex .∈ [K.subindex])

coord(K::SubsampledKSpaceSampling3D) = coord(K.kspace_sampling)[K.subindex,:]

struct SubsampledStructuredKSpaceSampling3D{T<:Real}<:AbstractStructuredKSpaceSampling3D{T}
    kspace_sampling::StructuredKSpaceSampling3D{T}
    subindex_phase_encoding::Union{Colon,AbstractVector{<:Integer}}
    subindex_readout::Union{Colon,AbstractVector{<:Integer}}
end

Base.getindex(K::StructuredKSpaceSampling3D{T}, subindex_phase_encoding::Union{Colon,AbstractVector{<:Integer}}, subindex_readout::Union{Colon,AbstractVector{<:Integer}}) where {T<:Real} = SubsampledStructuredKSpaceSampling3D{T}(K, subindex_phase_encoding, subindex_readout)
Base.getindex(K::SubsampledStructuredKSpaceSampling3D{T}, subindex_phase_encoding::Union{Colon,AbstractVector{<:Integer}}, subindex_readout::Union{Colon,AbstractVector{<:Integer}}) where {T<:Real} = K.kspace_sampling[K.subindex_phase_encoding[subindex_phase_encoding], K.subindex_readout[subindex_readout]]

isa_subsampling(Kq::SubsampledStructuredKSpaceSampling3D{T}, K::StructuredKSpaceSampling3D{T}) where {T<:Real} = (Kq.kspace_sampling == K)
isa_subsampling(Kq::SubsampledStructuredKSpaceSampling3D{T}, K::SubsampledStructuredKSpaceSampling3D{T}) where {T<:Real} = (Kq.kspace_sampling == K.kspace_sampling) && all(Kq.subindex_phase_encoding .∈ [K.subindex_phase_encoding]) && all(Kq.subindex_readout .∈ [K.subindex_readout])

coord_phase_encoding(K::SubsampledStructuredKSpaceSampling3D) = coord_phase_encoding(K.kspace_sampling)[K.subindex_phase_encoding,:]
coord_readout(K::SubsampledStructuredKSpaceSampling3D) = coord_readout(K.kspace_sampling)[K.subindex_readout]

dims_permutation(K::SubsampledStructuredKSpaceSampling3D) = dims_permutation(K.kspace_sampling)

struct SubsampledCartesianStructuredKSpaceSampling3D{T<:Real}<:AbstractStructuredKSpaceSampling3D{T}
    kspace_sampling::CartesianStructuredKSpaceSampling3D{T}
    subindex_phase_encoding::Union{Colon,AbstractVector{<:Integer}}
    subindex_readout::Union{Colon,AbstractVector{<:Integer}}
end

Base.getindex(K::CartesianStructuredKSpaceSampling3D{T}, subindex_phase_encoding::Union{Colon,AbstractVector{<:Integer}}, subindex_readout::Union{Colon,AbstractVector{<:Integer}}) where {T<:Real} = SubsampledCartesianStructuredKSpaceSampling3D{T}(K, subindex_phase_encoding, subindex_readout)
Base.getindex(K::SubsampledCartesianStructuredKSpaceSampling3D{T}, subindex_phase_encoding::Union{Colon,AbstractVector{<:Integer}}, subindex_readout::Union{Colon,AbstractVector{<:Integer}}) where {T<:Real} = K.kspace_sampling[K.subindex_phase_encoding[subindex_phase_encoding], K.subindex_readout[subindex_readout]]

isa_subsampling(Kq::SubsampledCartesianStructuredKSpaceSampling3D{T}, K::CartesianStructuredKSpaceSampling3D{T}) where {T<:Real} = (Kq.kspace_sampling == K)
isa_subsampling(Kq::SubsampledCartesianStructuredKSpaceSampling3D{T}, K::SubsampledCartesianStructuredKSpaceSampling3D{T}) where {T<:Real} = (Kq.kspace_sampling == K.kspace_sampling) && all(Kq.subindex_phase_encoding .∈ [K.subindex_phase_encoding]) && all(Kq.subindex_readout .∈ [K.subindex_readout])

coord_phase_encoding(K::SubsampledCartesianStructuredKSpaceSampling3D) = coord_phase_encoding(K.kspace_sampling)[K.subindex_phase_encoding,:]
coord_readout(K::SubsampledCartesianStructuredKSpaceSampling3D) = coord_readout(K.kspace_sampling)[K.subindex_readout]

dims_permutation(K::SubsampledCartesianStructuredKSpaceSampling3D) = dims_permutation(K.kspace_sampling)