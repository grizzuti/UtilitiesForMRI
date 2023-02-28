# 2D interpolations routines

function rotation_3D_eval(X::CartesianSpatialGeometry{T}, u::AbstractArray{CT, 3}, φ::AbstractVector{T}) where {T<:Real, CT<:RealOrComplex{T}}

    Ru = deepcopy(u)
    rotation_2D3D_eval!(X, Ru, (1, 2), φ[1])
    rotation_2D3D_eval!(X, Ru, (1, 3), φ[2])
    rotation_2D3D_eval!(X, Ru, (2, 3), φ[3])
    return Ru

end

rotation_2D3D_eval(X::CartesianSpatialGeometry{T}, u::AbstractArray{CT, 3}, plane_dims::NTuple{2, Integer}, φ::T) where {T<:Real, CT<:RealOrComplex{T}} = rotation_2D3D_eval!(X, deepcopy(u), plane_dims, φ)

function rotation_2D3D_eval!(X::CartesianSpatialGeometry{T}, u::AbstractArray{CT, 3}, plane_dims::NTuple{2, Integer}, φ::T) where {T<:Real, CT<:RealOrComplex{T}}

    # Geometric info
    gpu(x) = u isa CuArray ? convert(CuArray, x) : x
    x1 = gpu(coord(X, plane_dims[1])); x2 = gpu(coord(X, plane_dims[2]))
    x1 = reshape(x1, :, 1, 1); x2 = reshape(x2, 1, 1, :) ### needed to ensure non-sequential fft dims!
    k1 = gpu(k_coord(X, plane_dims[1])); k2 = gpu(k_coord(X, plane_dims[2]))
    k1 = reshape(k1, :, 1, 1); k2 = reshape(k2, 1, 1, :) ### needed to ensure non-sequential fft dims!

    # Permutation
    perm = (plane_dims[1], orth_dim(plane_dims), plane_dims[2]) ### needed to ensure non-sequential fft dims!
    n_perm = X.nsamples[[invperm(perm)...]]
    Ru_perm = similar(u, n_perm)
    temp_alloc = permutedims(u, invperm(perm))

    # Shearing (1)
    fft!(temp_alloc, 3)
    fftshift!(Ru_perm, temp_alloc, 3)
    phase_shift = exp.(-1im*tan(φ/2)*k2.*x1)
    Ru_perm .*= phase_shift
    fftshift!(temp_alloc, Ru_perm, 3)
    ifft!(temp_alloc, 3)

    # Shearing (2)
    fft!(temp_alloc, 1)
    fftshift!(Ru_perm, temp_alloc, 1)
    phase_shift .= exp.(1im*sin(φ)*k1.*x2)
    Ru_perm .*= phase_shift
    fftshift!(temp_alloc, Ru_perm, 1)
    ifft!(temp_alloc, 1)

    # Shearing (3)
    fft!(temp_alloc, 3)
    fftshift!(Ru_perm, temp_alloc, 3)
    phase_shift .= exp.(-1im*tan(φ/2)*k2.*x1)
    Ru_perm .*= phase_shift
    fftshift!(temp_alloc, Ru_perm, 3)
    ifft!(temp_alloc, 3)

    # Permutation
    return permutedims!(u, temp_alloc, perm)

end

function orth_dim(dims::NTuple{2,Integer})
    ((dims == (1, 2)) || (dims == (2, 1))) && (return 3)
    ((dims == (1, 3)) || (dims == (3, 1))) && (return 2)
    ((dims == (2, 3)) || (dims == (3, 2))) && (return 1)
end