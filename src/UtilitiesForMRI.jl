module UtilitiesForMRI

using LinearAlgebra, SparseArrays, AbstractLinearOperators, FINUFFT, FFTW, PyPlot, ImageFiltering, Dierckx, ImageQualityIndexes, CUDA
import CUDA.CUSPARSE: sparse

const RealOrComplex{T<:Real} = Union{T,Complex{T}}

include("./abstract_types.jl")
include("./spatial_geometry.jl")
include("./kspace_geometry_3D.jl")
include("./kspace_geometry_2DMS.jl")
include("./scaling_utils.jl")
include("./translations.jl")
include("./rotations.jl")
include("./nfft.jl")
include("./motion_parameter_utils.jl")
include("./plotting_utils.jl")
include("./imagequality_utils.jl")

end