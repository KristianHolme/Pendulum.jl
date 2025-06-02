using ClassicControlEnvironments
using Test

@testset "ClassicControlEnvironments.jl" begin
    include("test_pendulum.jl")
    include("test_mountaincar.jl")
end
