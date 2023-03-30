var documenterSearchIndex = {"docs":
[{"location":"functions/#Main-functions","page":"Main functions","title":"Main functions","text":"","category":"section"},{"location":"functions/#Spatial-geometry-utilities","page":"Main functions","title":"Spatial geometry utilities","text":"","category":"section"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"These are the main utilities to build spatial discretization objects.","category":"page"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"spatial_geometry(field_of_view::NTuple{3,T}, nsamples::NTuple{3,Integer}; origin::NTuple{3,T}=field_of_view./2) where {T<:Real}","category":"page"},{"location":"functions/#UtilitiesForMRI.spatial_geometry-Union{Tuple{T}, Tuple{Tuple{T, T, T}, Tuple{Integer, Integer, Integer}}} where T<:Real","page":"Main functions","title":"UtilitiesForMRI.spatial_geometry","text":"spatial_geometry(field_of_view::NTuple{3,T}, nsamples::NTuple{3,Integer};\n                 origin::NTuple{3,T}=field_of_view./2) where {T<:Real}\n\nReturn a 3D spatial geometry object that summarizes a Cartesian spatial discretization. Must specify domain size and number of samples per dimension. The origin of the domain can be changed by setting the keyword origin.\n\nExample:\n\nX = spatial_geometry((1f0, 1f0, 1f0), (64, 64, 64); origin=(0f0, 0f0, 0f0))\n\n\n\n\n\n","category":"method"},{"location":"functions/#k-space-geometry-utilities","page":"Main functions","title":"k-space geometry utilities","text":"","category":"section"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"These are the main utilities to specify k-space acquisition trajectories, where the ordering is typically dictated by the order of acquisition (e.g., a proxy for time).","category":"page"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"kspace_sampling(permutation_dims::NTuple{3,Integer}, coord_phase_encoding::AbstractArray{T,2}, coord_readout::AbstractVector{T}) where {T<:Real}","category":"page"},{"location":"functions/#UtilitiesForMRI.kspace_sampling-Union{Tuple{T}, Tuple{Tuple{Integer, Integer, Integer}, AbstractMatrix{T}, AbstractVector{T}}} where T<:Real","page":"Main functions","title":"UtilitiesForMRI.kspace_sampling","text":"kspace_sampling(permutation_dims::NTuple{3,Integer},\n                coord_phase_encoding::AbstractArray{T,2},\n                coord_readout::AbstractVector{T}) where {T<:Real}\n\nReturns a 3D k-space trajectory that is (time-wise) ordered by phase-encoding plane coordinates. The plane is orthogonal to a specific coordinate direction (e.g. x, y, or z). The input permutation_dims specifies the phase-encoding plane dimensions (in this order) and readout dimension. The coord_phase_encoding should be an array of size n_ttimes 3 and represents the phase-encoding plane coordinate. Similarly, the readout coordinate coord_readout should be an array of size n_k.\n\n\n\n\n\n","category":"method"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"kspace_sampling(X::CartesianSpatialGeometry{T}, phase_encoding_dims::NTuple{2,Integer}; phase_encode_sampling::Union{Nothing,AbstractVector{<:Integer}}=nothing, readout_sampling::Union{Nothing,AbstractVector{<:Integer}}=nothing) where {T<:Real}","category":"page"},{"location":"functions/#UtilitiesForMRI.kspace_sampling-Union{Tuple{T}, Tuple{CartesianSpatialGeometry{T}, Tuple{Integer, Integer}}} where T<:Real","page":"Main functions","title":"UtilitiesForMRI.kspace_sampling","text":"kspace_sampling(X::CartesianSpatialGeometry{T},\n                phase_encoding_dims::NTuple{2,Integer};\n                phase_encode_sampling::Union{Nothing,AbstractVector{<:Integer}}=nothing,\n                readout_sampling::Union{Nothing,AbstractVector{<:Integer}}=nothing) where {T<:Real}\n\nReturns a 3D Cartesian k-space trajectory that is (time-wise) ordered by phase-encoding plane coordinates. The plane is orthogonal to a specific coordinate direction (e.g. x, y, or z). The spatial discretization is determined by the CartesianSpatialGeometry object X. The phase_encoding_dims indicate the phase-encoding plane dimensions. Subsampling of the full Cartesian k-space is obtained with the optional keyword inputs phase_encode_sampling and readout_sampling.\n\n\n\n\n\n","category":"method"},{"location":"functions/#Fourier-transform-utilities","page":"Main functions","title":"Fourier-transform utilities","text":"","category":"section"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"In order to manipulate the non-uniform Fourier operator based on rigid-motion perturbation, see Section Getting started . ","category":"page"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"nfft_linop(X::CartesianSpatialGeometry{T}, K::UtilitiesForMRI.AbstractStructuredKSpaceSampling{T}; norm_constant::T=1/T(sqrt(prod(X.nsamples))), tol::T=T(1e-6)) where {T<:Real}","category":"page"},{"location":"functions/#UtilitiesForMRI.nfft_linop-Union{Tuple{T}, Tuple{CartesianSpatialGeometry{T}, UtilitiesForMRI.AbstractStructuredKSpaceSampling{T}}} where T<:Real","page":"Main functions","title":"UtilitiesForMRI.nfft_linop","text":"nfft_linop(X::CartesianSpatialGeometry, K::StructuredKSpaceSampling)\n\nReturn the non-uniform Fourier transform as a linear operator for a specified Cartesian spatial discretization X and a k-space trajectory K.\n\nExample\n\nX = spatial_geometry((1f0, 1f0, 1f0), (32, 32, 32))\nK = kspace_sampling(X, (1, 2))\nF = nfft_linop(X, K)\nu = randn(ComplexF32, X.nsamples)\nd = F*u  # evaluation\n    F'*d # adjoint\n\n\n\n\n\n","category":"method"},{"location":"functions/#Resampling-utilities","page":"Main functions","title":"Resampling utilities","text":"","category":"section"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"We list the main functionalities for subsampling/upsampling of spatial geometries, k-space geometries, and 3D images:","category":"page"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"resample(X::CartesianSpatialGeometry, n::NTuple{3,Integer})","category":"page"},{"location":"functions/#UtilitiesForMRI.resample-Tuple{CartesianSpatialGeometry, Tuple{Integer, Integer, Integer}}","page":"Main functions","title":"UtilitiesForMRI.resample","text":"resample(X::CartesianSpatialGeometry, n::NTuple{3,Integer})\n\nUp/down-sampling of Cartesian discretization geometry. The field of view and origin of X are kept the same, while the sampling is changed according to n.\n\n\n\n\n\n","category":"method"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"subsample(K::UtilitiesForMRI.AbstractStructuredKSpaceSampling{T}, k_max::Union{T,NTuple{3,T}}; radial::Bool=false, also_readout::Bool=true) where {T<:Real}","category":"page"},{"location":"functions/#UtilitiesForMRI.subsample-Union{Tuple{T}, Tuple{UtilitiesForMRI.AbstractStructuredKSpaceSampling{T}, Union{Tuple{T, T, T}, T}}} where T<:Real","page":"Main functions","title":"UtilitiesForMRI.subsample","text":"subsample(K, k_max::Union{T,NTuple{3,T}}; radial::Bool=false)\n\nDown-sampling of a k-space trajectory. The output is a subset of the original k-space trajectory such that the k-space coordinates (k_1k_2k_3) are:     - k_ile k_mathrmmaxi (if radial=false), or     - mathbfkle k_mathrmmax (if radial=true). The ordering of the subsampled trajectory is inherited from the original trajectory.\n\n\n\n\n\n","category":"method"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"subsample(K::UtilitiesForMRI.AbstractStructuredKSpaceSampling{T}, X::CartesianSpatialGeometry{T}; radial::Bool=false, also_readout::Bool=true) where {T<:Real}","category":"page"},{"location":"functions/#UtilitiesForMRI.subsample-Union{Tuple{T}, Tuple{UtilitiesForMRI.AbstractStructuredKSpaceSampling{T}, CartesianSpatialGeometry{T}}} where T<:Real","page":"Main functions","title":"UtilitiesForMRI.subsample","text":"subsample(K, X::CartesianSpatialGeometry;\n          radial::Bool=false, also_readout::Bool=true)\n\nDown-sampling of a k-space trajectory, similarly to subsample. The maximum cutoff frequency is inferred from the Nyquist frequency of a Cartesian spatial geometry X.\n\n\n\n\n\n","category":"method"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"subsample(K::StructuredKSpaceSampling{T}, d::AbstractArray{CT,2}, Kq::SubsampledStructuredKSpaceSampling{T}; norm_constant::Union{Nothing,T}=nothing, damping_factor::Union{T,Nothing}=nothing) where {T<:Real,CT<:RealOrComplex{T}}","category":"page"},{"location":"functions/#UtilitiesForMRI.subsample-Union{Tuple{CT}, Tuple{T}, Tuple{StructuredKSpaceSampling{T}, AbstractMatrix{CT}, SubsampledStructuredKSpaceSampling{T}}} where {T<:Real, CT<:Union{Complex{T}, T}}","page":"Main functions","title":"UtilitiesForMRI.subsample","text":"subsample(K, d::AbstractArray{<:Complex,2}, Kq; norm_constant=nothing)\n\nDown-sampling of a k-space data array d associated to a k-space trajectory K, e.g. d_i=d(\\mathbf{k}_i). The output is a subset of the original data array, which is associated to the down-sampled k-space Kq (obtained, for example, via subsample). The keyword norm_constant allows the rescaling of the down-sampled data.\n\n\n\n\n\n","category":"method"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"subsample(K::CartesianStructuredKSpaceSampling{T}, d::AbstractArray{CT,2}, Kq::SubsampledCartesianStructuredKSpaceSampling{T}; norm_constant::Union{Nothing,T}=nothing, damping_factor::Union{T,Nothing}=nothing) where {T<:Real,CT<:RealOrComplex{T}}","category":"page"},{"location":"functions/#UtilitiesForMRI.subsample-Union{Tuple{CT}, Tuple{T}, Tuple{CartesianStructuredKSpaceSampling{T}, AbstractMatrix{CT}, SubsampledCartesianStructuredKSpaceSampling{T}}} where {T<:Real, CT<:Union{Complex{T}, T}}","page":"Main functions","title":"UtilitiesForMRI.subsample","text":"subsample(K, d::AbstractArray{<:Complex,2}, Kq; norm_constant=nothing)\n\nDown-sampling of a k-space data array d associated to a k-space trajectory K, e.g. d_i=d(\\mathbf{k}_i). The output is a subset of the original data array, which is associated to the down-sampled k-space Kq (obtained, for example, via subsample). The keyword norm_constant allows the rescaling of the down-sampled data.\n\n\n\n\n\n","category":"method"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"resample(u::AbstractArray{CT,3}, n_scale::NTuple{3,Integer}; damping_factor::Union{T,Nothing}=nothing) where {T<:Real,CT<:RealOrComplex{T}}","category":"page"},{"location":"functions/#UtilitiesForMRI.resample-Union{Tuple{CT}, Tuple{T}, Tuple{AbstractArray{CT, 3}, Tuple{Integer, Integer, Integer}}} where {T<:Real, CT<:Union{Complex{T}, T}}","page":"Main functions","title":"UtilitiesForMRI.resample","text":"resample(u, n_scale::NTuple{3,Integer}; damping_factor=nothing)\n\nResampling of spatial array u. The underlying field of view represented by u is maintained, while the original sampling rate n=size(u) is changed to n_scale.\n\n\n\n\n\n","category":"method"},{"location":"functions/#Image-quality-metrics","page":"Main functions","title":"Image-quality metrics","text":"","category":"section"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"Convenience functions to compute slice-based PSNR and SSIM metrics for 3D images (these relies on the package ImageQualityMetrics ):","category":"page"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"psnr(u_noisy::AbstractArray{T,3}, u_ref::AbstractArray{T,3}; slices::Union{FullVolume,NTuple{N,VolumeSlice}}=full_volume(), orientation::Orientation=standard_orientation()) where {T<:Real,N}","category":"page"},{"location":"functions/#UtilitiesForMRI.psnr-Union{Tuple{N}, Tuple{T}, Tuple{AbstractArray{T, 3}, AbstractArray{T, 3}}} where {T<:Real, N}","page":"Main functions","title":"UtilitiesForMRI.psnr","text":"psnr(u_noisy::AbstractArray{T,3}, u_ref::AbstractArray{T,3};\n     slices::Union{FullVolume,NTuple{N,VolumeSlice}}=full_volume(),\n     orientation::Orientation=standard_orientation()) where {T<:Real,N}\n\nCompute 2D/3D peak signal-to-noise ratio for the indicated 2D slices of a 3D array or full volume. The optional keyword slices indicates the 2D slices in object (see volume_slice), according to the 3D orientation (see orientation).\n\n\n\n\n\n","category":"method"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"ssim(u_noisy::AbstractArray{T,3}, u_ref::AbstractArray{T,3}; slices::Union{FullVolume,NTuple{N,VolumeSlice}}=full_volume(), orientation::Orientation=standard_orientation()) where {T<:Real,N}","category":"page"},{"location":"functions/#UtilitiesForMRI.ssim-Union{Tuple{N}, Tuple{T}, Tuple{AbstractArray{T, 3}, AbstractArray{T, 3}}} where {T<:Real, N}","page":"Main functions","title":"UtilitiesForMRI.ssim","text":"ssim(u_noisy::AbstractArray{T,3}, u_ref::AbstractArray{T,3};\n     slices::Union{FullVolume,NTuple{N,VolumeSlice}}=full_volume,\n     orientation::Orientation=standard_orientation()) where {T<:Real,N}\n\nCompute 2D/3D structural similarity index for the indicated 2D slices of a 3D array or full volume. The optional keyword slices indicates the 2D slices in object (see volume_slice), according to the 3D orientation (see orientation).\n\n\n\n\n\n","category":"method"},{"location":"functions/#Visualization-tools","page":"Main functions","title":"Visualization tools","text":"","category":"section"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"orientation(perm::NTuple{3,Integer}; reverse::NTuple{3,Bool}=(false,false,false))","category":"page"},{"location":"functions/#UtilitiesForMRI.orientation-Tuple{Tuple{Integer, Integer, Integer}}","page":"Main functions","title":"UtilitiesForMRI.orientation","text":"orientation(perm::NTuple{3,Integer}; reverse::NTuple{3,Bool}=(false,false,false))\n\nDefines an orientation for a 3D image. It is mostly useful for standardizing plotting utilities. For example it can be used to reorder a 3D image such that in radiological terms, after reordering:     - 1st dimension = left-right     - 2nd dimension = posterior-anterior     - 3rd dimension = inferior-superior\n\nThe input perm determines an ordering for the 3D dimensions, while the keyword reverse specifies which dimensions are negatively oriented.\n\n\n\n\n\n","category":"method"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"standard_orientation()","category":"page"},{"location":"functions/#UtilitiesForMRI.standard_orientation-Tuple{}","page":"Main functions","title":"UtilitiesForMRI.standard_orientation","text":"standard_orientation()\n\nReturns the standard orientation (no reordering).\n\n\n\n\n\n","category":"method"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"volume_slice(dim, n; window=nothing)","category":"page"},{"location":"functions/#UtilitiesForMRI.volume_slice-Tuple{Any, Any}","page":"Main functions","title":"UtilitiesForMRI.volume_slice","text":"volume_slice(dim::Integer, n::Integer; window=nothing)\n\nSpecifies a 2D slice with respect to a 3D image: dim is the dimension orthogonal to the 2D slice, n is the position of the slice, and window indicates a portion of the 2D slice.\n\nImportant: volume slices are always defined with respect to a given orientation of the 3D object.\n\n\n\n\n\n","category":"method"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"select(u::AbstractArray{T,3}, slice::VolumeSlice; orientation::Orientation=standard_orientation()) where {T<:Real}","category":"page"},{"location":"functions/#UtilitiesForMRI.select-Union{Tuple{T}, Tuple{AbstractArray{T, 3}, VolumeSlice}} where T<:Real","page":"Main functions","title":"UtilitiesForMRI.select","text":"select(u::AbstractArray{T,3}, slice::VolumeSlice; orientation::Orientation=standard_orientation()) where {T<:Real}\n\nReturns the specified 2D slice of a 3D image (according to a specified orientation)\n\n\n\n\n\n","category":"method"},{"location":"functions/","page":"Main functions","title":"Main functions","text":"plot_volume_slices(u::AbstractArray{T,3};\n    slices::Union{Nothing,NTuple{N,VolumeSlice}}=nothing,\n    spatial_geometry::Union{Nothing,CartesianSpatialGeometry{T}}=nothing,\n    cmap::String=\"gray\",\n    vmin::Union{Nothing,Real}=nothing, vmax::Union{Nothing,Real}=nothing,\n    xlabel::Union{Nothing,AbstractString}=nothing, ylabel::Union{Nothing,AbstractString}=nothing,\n    cbar_label::Union{Nothing,AbstractString}=nothing,\n    title::Union{Nothing,AbstractString}=nothing,\n    savefile::Union{Nothing,String}=nothing,\n    orientation::Orientation=standard_orientation()) where {T<:Real,N}","category":"page"},{"location":"functions/#UtilitiesForMRI.plot_volume_slices-Union{Tuple{AbstractArray{T, 3}}, Tuple{N}, Tuple{T}} where {T<:Real, N}","page":"Main functions","title":"UtilitiesForMRI.plot_volume_slices","text":"plot_volume_slices(u::AbstractArray{T,3};\n                   slices::Union{Nothing,NTuple{N,VolumeSlice}}=nothing,\n                   spatial_geometry::Union{Nothing,CartesianSpatialGeometry{T}}=nothing,\n                   cmap::String=\"gray\",\n                   vmin::Union{Nothing,Real}=nothing, vmax::Union{Nothing,Real}=nothing,\n                   xlabel::Union{Nothing,AbstractString}=nothing, ylabel::Union{Nothing,AbstractString}=nothing,\n                   cbar_label::Union{Nothing,AbstractString}=nothing,\n                   title::Union{Nothing,AbstractString}=nothing,\n                   savefile::Union{Nothing,String}=nothing,\n                   orientation::Orientation=standard_orientation()) where {T<:Real,N}\n\nPlot 2D slices of a given 3D image.\n\n\n\n\n\n","category":"method"},{"location":"examples/#Getting-started","page":"Getting started","title":"Getting started","text":"","category":"section"},{"location":"examples/","page":"Getting started","title":"Getting started","text":"We provide a simple example to perform a rigid-body motion using the tools provided by UtilitiesForMRI.","category":"page"},{"location":"examples/","page":"Getting started","title":"Getting started","text":"For starters, let's make sure that all the needed packages are installed! Please, follow the instructions in Section Installation instructions. For this tutorial, we also need PyPlot. To install, Type ] in the Julia REPL and","category":"page"},{"location":"examples/","page":"Getting started","title":"Getting started","text":"@(v1.8) pkg> add PyPlot","category":"page"},{"location":"examples/","page":"Getting started","title":"Getting started","text":"Here, we use PyPlot for image visualization, but many other packages may fit the bill.","category":"page"},{"location":"examples/","page":"Getting started","title":"Getting started","text":"To load the relevant modules:","category":"page"},{"location":"examples/","page":"Getting started","title":"Getting started","text":"# Package load\nusing UtilitiesForMRI, LinearAlgebra, PyPlot","category":"page"},{"location":"examples/","page":"Getting started","title":"Getting started","text":"Let's define a Cartesian spatial discretization for a 3D image:","category":"page"},{"location":"examples/","page":"Getting started","title":"Getting started","text":"# Cartesian domain\nn = (64, 64, 64)\nfov = (1.0, 1.0, 1.0)\norigin = (0.5, 0.5, 0.5)\nX = spatial_geometry(fov, n; origin=origin)","category":"page"},{"location":"examples/","page":"Getting started","title":"Getting started","text":"We can also set a simple k-space trajectory:","category":"page"},{"location":"examples/","page":"Getting started","title":"Getting started","text":"# Cartesian sampling in k-space\nphase_encoding_dims = (1,2)\nK = kspace_sampling(X, phase_encoding_dims)\nnt, nk = size(K)","category":"page"},{"location":"examples/","page":"Getting started","title":"Getting started","text":"The Fourier operator for 3D images based on the X discretization and K sampling is:","category":"page"},{"location":"examples/","page":"Getting started","title":"Getting started","text":"# Fourier operator\nF = nfft_linop(X, K)","category":"page"},{"location":"examples/","page":"Getting started","title":"Getting started","text":"Let's assume we want to perform a rigid motion for a certain image:","category":"page"},{"location":"examples/","page":"Getting started","title":"Getting started","text":"# Rigid-body perturbation\nθ = zeros(Float64, nt, 6)\nθ[:, 1] .= 0       # x translation\nθ[:, 2] .= 0       # y translation\nθ[:, 3] .= 0       # z translation\nθ[:, 4] .= 2*pi/10 # xy rotation\nθ[:, 5] .= 0       # xz rotation\nθ[:, 6] .= 0       # yz rotation\n\n# 3D image\nu = zeros(ComplexF64, n); u[33-10:33+10, 33-10:33+10, 33-10:33+10] .= 1","category":"page"},{"location":"examples/","page":"Getting started","title":"Getting started","text":"This can be easily done by evaluating the rigid-motion perturbed Fourier transform, and applying the adjoint of the conventional Fourier transform, e.g.:","category":"page"},{"location":"examples/","page":"Getting started","title":"Getting started","text":"# Rigid-body motion\nu_rbm = F'*F(θ)*u","category":"page"},{"location":"examples/","page":"Getting started","title":"Getting started","text":"For plotting:","category":"page"},{"location":"examples/","page":"Getting started","title":"Getting started","text":"# Plotting\nfigure()\nimshow(abs.(u_rbm[:,:,33]); vmin=0, vmax=1)","category":"page"},{"location":"installation/#Installation-instructions","page":"Installation","title":"Installation instructions","text":"","category":"section"},{"location":"installation/","page":"Installation","title":"Installation","text":"In the Julia REPL, simply type ] and","category":"page"},{"location":"installation/","page":"Installation","title":"Installation","text":"(@v1.8) pkg> add https://github.com/grizzuti/AbstractLinearOperators.git, add https://github.com/grizzuti/UtilitiesForMRI.git","category":"page"},{"location":"installation/","page":"Installation","title":"Installation","text":"The package AbstractLinearOperators has to be explicitly installed since it is currently unregistered.","category":"page"},{"location":"#Introduction","page":"Introduction","title":"Introduction","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"This package implements some basic utilities for MRI, for example related to the definition of the:","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Cartesian x-space discretization of a given field of view;\nordered acquisition trajectory in k-space;\nFourier transform for several x/k-space specifications.","category":"page"},{"location":"","page":"Introduction","title":"Introduction","text":"Notably, UtilitiesForMRI allows the perturbation of the Fourier transform F with respect to some time-dependent rigid body motion theta. Furthermore, the mapping thetamapsto F(theta)mathbfu is differentiable, which can be useful for motion correction (see [1]).","category":"page"},{"location":"#Relevant-publications","page":"Introduction","title":"Relevant publications","text":"","category":"section"},{"location":"","page":"Introduction","title":"Introduction","text":"Rizzuti, G., Sbrizzi, A., and van Leeuwen, T., (2022). Joint Retrospective Motion Correction and Reconstruction for Brain MRI With a Reference Contrast, IEEE Transaction on Computational Imaging, 8, 490-504, doi:10.1109/TCI.2022.3183383","category":"page"}]
}
