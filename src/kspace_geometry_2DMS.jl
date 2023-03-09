# 2D-multislice k-space geometry utilities

export KSpaceSampling2DMS, SubsampledKSpaceSampling2DMS, StructuredKSpaceSampling2DMS, SubsampledStructuredKSpaceSampling2DMS, CartesianStructuredKSpaceSampling2DMS, SubsampledCartesianStructuredKSpaceSampling2DMS


## k-space sampling (= ordered wave-number coordinates)

struct KSpaceSampling2DMS{T}<:AbstractKSpaceSampling2DMS{T}
    slice_plane::Integer # plane-orthogonal direction
    slice_coord::AbstractVector{<:AbstractVector{T}} # ordered slice coordinates
    k_coord::AbstractArray{T,2} # k-space coordinates size=(nt,2)
end

kspace_sampling(slice_plane::Integer, slice_coord::AbstractVector{<:AbstractVector{T}}, k_coord::AbstractArray{T,2}) where {T<:Real} = KSpaceSampling2DMS{T}(slice_plane, slice_coord, k_coord)

coord(K::KSpaceSampling2DMS{T}) where {T<:Real} = K.k_coord
slice_plane(K::KSpaceSampling2DMS{T}) where {T<:Real} = K.slice_plane
slice_coord(K::KSpaceSampling2DMS{T}) where {T<:Real} = K.slice_coord


## Structured k-space sampling

struct StructuredKSpaceSampling2DMS{T}<:AbstractStructuredKSpaceSampling2DMS{T}
    slice_plane::Integer # plane-orthogonal direction
    slice_coord::AbstractVector{<:AbstractVector{T}} # ordered slice coordinates
    phase_encoding_dim::Integer
    coord_phase_encoding::AbstractVector{T} # k-space coordinates size=(nt,)
    coord_readout::AbstractVector{T} # k-space coordinates size=(nk,)
end

kspace_sampling(slice_plane::Integer, slice_coord::AbstractVector{<:AbstractVector{T}}, phase_encoding_dim::Integer, coord_phase_encoding::AbstractVector{T}, coord_readout::AbstractVector{T}) where {T<:Real} = StructuredKSpaceSampling2DMS{T}(slice_plane, slice_coord, phase_encoding_dim, coord_phase_encoding, coord_readout)

coord_phase_encoding(K::StructuredKSpaceSampling2DMS) = K.coord_phase_encoding
coord_readout(K::StructuredKSpaceSampling2DMS) = K.coord_readout

perm(K::StructuredKSpaceSampling2DMS) = (K.phase_encoding_dim == 1) ? (1, 2) : (2, 1)

function coord(K::AbstractStructuredKSpaceSampling2DMS)
    k_pe, k_r = coord_phase_encoding(K), coord_readout(K)
    perm = (K.phase_encoding_dim == 1)
    k_coord = cat(repeat(reshape(k_pe, :, 1, 1); outer=(1, length(k_r), 1)),
                  repeat(reshape(k_r,  1, :, 1); outer=(length(k_pe), 1, 1)); dims=3)[:, :, invperm(perm(K))]
    return k_coord
end

Base.size(K::AbstractStructuredKSpaceSampling2DMS) = (length(coord_phase_encoding(K)),length(coord_readout(K)))


## Cartesian structured k-space sampling

struct CartesianStructuredKSpaceSampling2DMS{T}<:AbstractStructuredKSpaceSampling2DMS{T}
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
    return CartesianStructuredKSpaceSampling2DMS{T}(X, tuple(permutation_dims...), phase_encode_sampling, readout_sampling)

end

function coord(K::CartesianStructuredKSpaceSampling2DMS{T}) where {T<:Real}
    nt, nk = size(K)
    perm = dims_permutation(K)
    k_pe, k_r = coord_phase_encoding(K), coord_readout(K)
    k_coordinates = cat(repeat(reshape(k_pe, nt,1,2); outer=(1,nk,1)), repeat(reshape(k_r, 1,nk,1); outer=(nt,1,1)); dims=3)[:,:,invperm(perm)]
    return k_coordinates
end

function coord_phase_encoding(K::CartesianStructuredKSpaceSampling2DMS)
    k_pe1, k_pe2 = k_coord(K.spatial_geometry; mesh=false)[dims_permutation(K)][1:2]
    n_pe1 = length(k_pe1); n_pe2 = length(k_pe2)
    return reshape(cat(repeat(reshape(k_pe1,:,1); outer=(1,n_pe2)), repeat(reshape(k_pe2,1,:); outer=(n_pe1,1)); dims=3),:,2)[K.idx_phase_encoding,:]
end

function coord_readout(K::CartesianStructuredKSpaceSampling2DMS)
    k_r = k_coord(K.spatial_geometry; mesh=false)[dims_permutation(K)][3]
    return k_r[K.idx_readout]
end

Base.convert(::Type{StructuredKSpaceSampling2DMS{T}}, K::CartesianStructuredKSpaceSampling2DMS{T}) where {T<:Real} = kspace_sampling(K.permutation_dims, coord_phase_encoding(K), coord_readout(K))
Base.convert(::Type{StructuredKSpaceSampling2DMS}, K::CartesianStructuredKSpaceSampling2DMS{T}) where {T<:Real} = convert(StructuredKSpaceSampling{T}, K)


## Utils

dims_permutation(K::AbstractStructuredKSpaceSampling2DMS) = [K.permutation_dims...]

dims_permutation(phase_encoding_dims::NTuple{2,Integer}) = [phase_encoding_dims..., readout_dim(phase_encoding_dims)]

function readout_dim(phase_encoding::NTuple{2,Integer})
    ((phase_encoding == (1,2)) || (phase_encoding == (2,1))) && (readout = 3)
    ((phase_encoding == (1,3)) || (phase_encoding == (3,1))) && (readout = 2)
    ((phase_encoding == (2,3)) || (phase_encoding == (3,2))) && (readout = 1)
    return readout
end


## Sub-sampled k-space

struct SubsampledKSpaceSampling2DMS{T<:Real}<:AbstractKSpaceSampling2DMS{T}
    kspace_sampling::KSpaceSampling2DMS{T}
    subindex::Union{Colon,AbstractVector{<:Integer}}
end

Base.getindex(K::KSpaceSampling2DMS{T}, subindex::Union{Colon,AbstractVector{<:Integer}}) where {T<:Real} = SubsampledKSpaceSampling2DMS{T}(K, subindex)
Base.getindex(K::SubsampledKSpaceSampling2DMS{T}, subindex::Union{Colon,AbstractVector{<:Integer}}) where {T<:Real} = K.kspace_sampling[K.subindex[subindex]]

isa_subsampling(Kq::SubsampledKSpaceSampling2DMS{T}, K::KSpaceSampling2DMS{T}) where {T<:Real} = (Kq.kspace_sampling == K)
isa_subsampling(Kq::SubsampledKSpaceSampling2DMS{T}, K::SubsampledKSpaceSampling2DMS{T}) where {T<:Real} = (Kq.kspace_sampling == K.kspace_sampling) && all(Kq.subindex .∈ [K.subindex])

coord(K::SubsampledKSpaceSampling2DMS) = coord(K.kspace_sampling)[K.subindex,:]

struct SubsampledStructuredKSpaceSampling2DMS{T<:Real}<:AbstractStructuredKSpaceSampling2DMS{T}
    kspace_sampling::StructuredKSpaceSampling2DMS{T}
    subindex_phase_encoding::Union{Colon,AbstractVector{<:Integer}}
    subindex_readout::Union{Colon,AbstractVector{<:Integer}}
end

Base.getindex(K::StructuredKSpaceSampling2DMS{T}, subindex_phase_encoding::Union{Colon,AbstractVector{<:Integer}}, subindex_readout::Union{Colon,AbstractVector{<:Integer}}) where {T<:Real} = SubsampledStructuredKSpaceSampling2DMS{T}(K, subindex_phase_encoding, subindex_readout)
Base.getindex(K::SubsampledStructuredKSpaceSampling2DMS{T}, subindex_phase_encoding::Union{Colon,AbstractVector{<:Integer}}, subindex_readout::Union{Colon,AbstractVector{<:Integer}}) where {T<:Real} = K.kspace_sampling[K.subindex_phase_encoding[subindex_phase_encoding], K.subindex_readout[subindex_readout]]

isa_subsampling(Kq::SubsampledStructuredKSpaceSampling2DMS{T}, K::StructuredKSpaceSampling2DMS{T}) where {T<:Real} = (Kq.kspace_sampling == K)
isa_subsampling(Kq::SubsampledStructuredKSpaceSampling2DMS{T}, K::SubsampledStructuredKSpaceSampling2DMS{T}) where {T<:Real} = (Kq.kspace_sampling == K.kspace_sampling) && all(Kq.subindex_phase_encoding .∈ [K.subindex_phase_encoding]) && all(Kq.subindex_readout .∈ [K.subindex_readout])

coord_phase_encoding(K::SubsampledStructuredKSpaceSampling2DMS) = coord_phase_encoding(K.kspace_sampling)[K.subindex_phase_encoding,:]
coord_readout(K::SubsampledStructuredKSpaceSampling2DMS) = coord_readout(K.kspace_sampling)[K.subindex_readout]

dims_permutation(K::SubsampledStructuredKSpaceSampling2DMS) = dims_permutation(K.kspace_sampling)

struct SubsampledCartesianStructuredKSpaceSampling2DMS{T<:Real}<:AbstractStructuredKSpaceSampling2DMS{T}
    kspace_sampling::CartesianStructuredKSpaceSampling2DMS{T}
    subindex_phase_encoding::Union{Colon,AbstractVector{<:Integer}}
    subindex_readout::Union{Colon,AbstractVector{<:Integer}}
end

Base.getindex(K::CartesianStructuredKSpaceSampling2DMS{T}, subindex_phase_encoding::Union{Colon,AbstractVector{<:Integer}}, subindex_readout::Union{Colon,AbstractVector{<:Integer}}) where {T<:Real} = SubsampledCartesianStructuredKSpaceSampling2DMS{T}(K, subindex_phase_encoding, subindex_readout)
Base.getindex(K::SubsampledCartesianStructuredKSpaceSampling2DMS{T}, subindex_phase_encoding::Union{Colon,AbstractVector{<:Integer}}, subindex_readout::Union{Colon,AbstractVector{<:Integer}}) where {T<:Real} = K.kspace_sampling[K.subindex_phase_encoding[subindex_phase_encoding], K.subindex_readout[subindex_readout]]

isa_subsampling(Kq::SubsampledCartesianStructuredKSpaceSampling2DMS{T}, K::CartesianStructuredKSpaceSampling2DMS{T}) where {T<:Real} = (Kq.kspace_sampling == K)
isa_subsampling(Kq::SubsampledCartesianStructuredKSpaceSampling2DMS{T}, K::SubsampledCartesianStructuredKSpaceSampling2DMS{T}) where {T<:Real} = (Kq.kspace_sampling == K.kspace_sampling) && all(Kq.subindex_phase_encoding .∈ [K.subindex_phase_encoding]) && all(Kq.subindex_readout .∈ [K.subindex_readout])

coord_phase_encoding(K::SubsampledCartesianStructuredKSpaceSampling2DMS) = coord_phase_encoding(K.kspace_sampling)[K.subindex_phase_encoding,:]
coord_readout(K::SubsampledCartesianStructuredKSpaceSampling2DMS) = coord_readout(K.kspace_sampling)[K.subindex_readout]

dims_permutation(K::SubsampledCartesianStructuredKSpaceSampling2DMS) = dims_permutation(K.kspace_sampling)