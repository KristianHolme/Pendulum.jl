@kwdef mutable struct PendulumProblem
    theta::Float32 = rand(Float32) * 2π - π # in [-π, π]
    velocity::Float32 = (rand(Float32) * 2.0f0 - 1.0f0) # in [-1, 1]
    torque::Float32 = 0.0f0
    mass::Float32 = 1.0f0
    length::Float32 = 1.0f0
    gravity::Float32 = 10f0
    dt::Float32 = 0.05f0
end

mutable struct PendulumEnv <: AbstractEnv
    problem::PendulumProblem
    action_space::Box{Float32}
    observation_space::Box{Float32}
    max_steps::Int
    step::Int
    rng::Random.AbstractRNG
    function PendulumEnv(; problem=nothing, max_steps::Int=200, rng::Random.AbstractRNG=Random.Xoshiro(), kwargs...)
        # Create a problem if not provided, using kwargs for its constructor
        if isnothing(problem)
            problem = PendulumProblem(; kwargs...)
        end

        action_space = Box{Float32}([-2.0f0], [2.0f0])
        observation_space = Box{Float32}([-1f0, -1f0, -8f0], [1f0, 1f0, 8f0])
        env = new(problem, action_space, observation_space, max_steps, 0, rng)
        return env
    end
end

function DRiL.reset!(env::PendulumEnv)
    # Use the environment's internal RNG
    reset!(env.problem, env.rng)
    env.step = 0
    nothing
end

function reset!(problem::PendulumProblem, rng::AbstractRNG)
    problem.theta = rand(rng, Float32) * 2π - π  # [-π, π] 
    problem.velocity = (rand(rng, Float32) * 2.0f0 - 1.0f0)  # [-1, 1] 
    problem.torque = 0.0f0
end

function pendulum_rewards(theta, velocity, torque)
    return -theta^2, -0.1f0 * velocity^2, -0.001f0 * torque^2
end

function reward(env::PendulumEnv)
    theta = env.problem.theta
    velocity = env.problem.velocity
    torque = env.problem.torque
    reward = sum(pendulum_rewards(theta, velocity, torque))
    return reward
end

function DRiL.act!(env::PendulumEnv, action::AbstractArray{Float32,1})
    DRiL.act!(env, action[1])
end

function DRiL.act!(env::PendulumEnv, action::Float32)
    pend = env.problem
    pend.torque = action
    g = pend.gravity
    m = pend.mass
    L = pend.length
    dt = pend.dt
    theta = pend.theta
    pend.velocity += ((3 * g / 2L) * sin(theta) + 3 / (m * L^2) * pend.torque) * dt
    pend.velocity = clamp(pend.velocity, -8.0f0, 8.0f0)
    pend.theta += pend.velocity * dt
    pend.theta = mod(pend.theta + π, 2π) - π
    env.step += 1
    return reward(env)
end

function DRiL.observe(env::PendulumEnv)
    x = cos(env.problem.theta)
    y = sin(env.problem.theta)
    return [x, y, env.problem.velocity]
end

DRiL.terminated(env::PendulumEnv) = false
DRiL.truncated(env::PendulumEnv) = env.step >= env.max_steps
DRiL.action_space(env::PendulumEnv) = env.action_space
DRiL.observation_space(env::PendulumEnv) = env.observation_space
DRiL.get_info(env::PendulumEnv) = Dict{String, Any}("step" => env.step)
