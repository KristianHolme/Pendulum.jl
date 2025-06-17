@kwdef mutable struct MountainCarProblem
    position::Float32 = rand(Float32)*0.2f0 - 0.6f0  # Car position, starts in [-0.6, -0.4]
    velocity::Float32 = 0.0f0   # Car velocity, in [-0.07, 0.07]
    force::Float32 = 0.0f0      # Applied force (action)
    power::Float32 = 0.0015f0   # Power multiplier for force
    gravity::Float32 = 0.0025f0 # Gravity constant
    goal_position::Float32 = 0.45f0  # Goal position (right hill)
    goal_velocity::Float32 = 0.0f0  # Goal velocity (can be modified)
    min_position::Float32 = -1.2f0
    max_position::Float32 = 0.6f0
    max_speed::Float32 = 0.07f0
end

# Abstract base type for both mountain car environments
abstract type AbstractMountainCarEnv <: AbstractEnv end

# Discrete Mountain Car Environment (0=left, 1=no force, 2=right)
mutable struct MountainCarEnv <: AbstractMountainCarEnv
    problem::MountainCarProblem
    action_space::Discrete
    observation_space::Box{Float32}
    max_steps::Int
    step::Int
    rng::Random.AbstractRNG

    function MountainCarEnv(; problem=nothing, max_steps::Int=999, rng::Random.AbstractRNG=Random.Xoshiro(), kwargs...)
        # Create a problem if not provided, using kwargs for its constructor
        if isnothing(problem)
            problem = MountainCarProblem(; kwargs...)
        end

        # Discrete action space: 0 = push left, 1 = no force, 2 = push right
        action_space = Discrete(3, 0)

        # Observation space: [position, velocity]
        observation_space = Box{Float32}(
            [problem.min_position, -problem.max_speed],
            [problem.max_position, problem.max_speed]
        )

        env = new(problem, action_space, observation_space, max_steps, 0, rng)
        return env
    end
end

# Continuous Mountain Car Environment (force in [-1, 1])
mutable struct MountainCarContinuousEnv <: AbstractMountainCarEnv
    problem::MountainCarProblem
    action_space::Box{Float32}
    observation_space::Box{Float32}
    max_steps::Int
    step::Int
    rng::Random.AbstractRNG

    function MountainCarContinuousEnv(; problem=nothing, max_steps::Int=999, rng::Random.AbstractRNG=Random.Xoshiro(), kwargs...)
        # Create a problem if not provided, using kwargs for its constructor
        if isnothing(problem)
            problem = MountainCarProblem(; kwargs...)
        end

        # Continuous action space: force in [-1, 1]
        action_space = Box{Float32}([-1.0f0], [1.0f0])

        # Observation space: [position, velocity]
        observation_space = Box{Float32}(
            [problem.min_position, -problem.max_speed],
            [problem.max_position, problem.max_speed]
        )

        env = new(problem, action_space, observation_space, max_steps, 0, rng)
        return env
    end
end

# Common methods for all mountain car environments

function DRiL.reset!(env::AbstractMountainCarEnv)
    # Use the environment's internal RNG
    reset!(env.problem, env.rng)
    env.step = 0
end

function reset!(problem::MountainCarProblem, rng::AbstractRNG)
    # Random position between -0.6 and -0.4
    problem.position = rand(rng, Float32) * 0.2f0 - 0.6f0  # [-0.6, -0.4]
    problem.velocity = 0.0f0
    problem.force = 0.0f0
    nothing
end

function reward(env::AbstractMountainCarEnv)
    # Reward structure for mountain car
    # Small penalty for each step to encourage reaching goal quickly
    # Bonus for reaching goal
    if env.problem.position >= env.problem.goal_position &&
       env.problem.velocity >= env.problem.goal_velocity
        return 100.0f0  # Large positive reward for reaching goal
    else
        # Small penalty per step + penalty for energy usage
        action_penalty = -0.1f0 * env.problem.force^2
        # step_penalty = -1.0f0
        return action_penalty
    end
end

# Shared physics update function
function update_mountain_car_physics!(problem::MountainCarProblem, force::Float32)
    problem.force = force

    # Update velocity based on force and gravity
    # velocity += force * power - cos(3 * position) * gravity
    velocity_change = force * problem.power - cos(3.0f0 * problem.position) * problem.gravity
    problem.velocity += velocity_change

    # Clamp velocity to bounds
    problem.velocity = clamp(problem.velocity, -problem.max_speed, problem.max_speed)

    # Update position
    problem.position += problem.velocity

    # Clamp position to bounds (with elastic collision at boundaries)
    if problem.position <= problem.min_position
        problem.position = problem.min_position
        problem.velocity = 0.0f0  # Stop at left boundary
    elseif problem.position >= problem.max_position
        problem.position = problem.max_position
        problem.velocity = 0.0f0  # Stop at right boundary (though this shouldn't happen in normal case)
    end

    nothing
end

# Discrete MountainCarEnv action methods
function DRiL.act!(env::MountainCarEnv, action::AbstractArray{Int,1})
    DRiL.act!(env, action[1])
end

function DRiL.act!(env::MountainCarEnv, action::Integer)
    # Convert discrete action to force: 0 -> -1, 1 -> 0, 2 -> +1
    force = action == 0 ? -1.0f0 : (action == 1 ? 0.0f0 : 1.0f0)
    
    update_mountain_car_physics!(env.problem, force)
    env.step += 1
    return reward(env)
end

# Continuous MountainCarContinuousEnv action methods
function DRiL.act!(env::MountainCarContinuousEnv, action::AbstractArray{Float32,1})
    DRiL.act!(env, action[1])
end

function DRiL.act!(env::MountainCarContinuousEnv, action::Float32)
    # Clamp action to valid range
    force = clamp(action, -1.0f0, 1.0f0)
    
    update_mountain_car_physics!(env.problem, force)
    env.step += 1
    return reward(env)
end

# Common interface methods
function DRiL.observe(env::AbstractMountainCarEnv)
    return [env.problem.position, env.problem.velocity]
end

function DRiL.terminated(env::AbstractMountainCarEnv)
    # Episode terminates when car reaches goal position with sufficient velocity
    return env.problem.position >= env.problem.goal_position &&
           env.problem.velocity >= env.problem.goal_velocity
end

DRiL.truncated(env::AbstractMountainCarEnv) = env.step >= env.max_steps
DRiL.action_space(env::AbstractMountainCarEnv) = env.action_space
DRiL.observation_space(env::AbstractMountainCarEnv) = env.observation_space
DRiL.get_info(env::AbstractMountainCarEnv) = Dict{String, Any}("step" => env.step, "position" => env.problem.position, "velocity" => env.problem.velocity)