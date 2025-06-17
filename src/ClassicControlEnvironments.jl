module ClassicControlEnvironments

using DRiL
using Random
using Reexport

include("utils.jl")
# Export the main environments
include("Pendulum.jl")
export PendulumEnv, PendulumProblem

include("MountainCar.jl")
export AbstractMountainCarEnv, MountainCarEnv, MountainCarContinuousEnv, MountainCarProblem

include("CartPole.jl")
export CartPoleEnv, CartPoleProblem

include("Acrobot.jl")
export AcrobotEnv, AcrobotProblem

function plot end
function live_viz end
function interactive_viz end
function plot_trajectory end
function plot_trajectory_interactive end
function animate_trajectory_video end
function plot_trajectory_phase_space end
export plot, live_viz, interactive_viz, plot_trajectory, plot_trajectory_interactive, animate_trajectory_video

# Placeholder exports for future environments
# export CartPoleEnv, MountainCarEnv, etc.

end