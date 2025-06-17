using ClassicControlEnvironments
using Test

@testset "ClassicControlEnvironments.jl" begin
    include("test_pendulum.jl")
    include("test_mountaincar.jl")
    include("test_acrobot.jl")
end
