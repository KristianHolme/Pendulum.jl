module ClassicalControl

using DRiL
using Random
using Reexport

# Export the main environments
include("Pendulum.jl")



function plot end
function live_viz end
function interactive_viz end
function plot_trajectory end
function plot_trajectory_interactive end
function animate_trajectory_video end


# Placeholder exports for future environments
# export CartPoleEnv, MountainCarEnv, etc.

end