@kwdef mutable struct CartPoleProblem
    # State variables
    x::Float32 = 0.0f0  # Cart position
    x_dot::Float32 = 0.0f0  # Cart velocity
    theta::Float32 = 0.0f0  # Pole angle (0 = upright, positive = clockwise)
    theta_dot::Float32 = 0.0f0  # Pole angular velocity
    force::Float32 = 0.0f0  # Applied force (action)
    
    # Environment parameters - matching Gymnasium CartPole-v1
    gravity::Float32 = 9.8f0
    masscart::Float32 = 1.0f0
    masspole::Float32 = 0.1f0
    total_mass::Float32 = masscart + masspole  # Derived parameter
    length::Float32 = 0.5f0  # Half-pole length (pole center of mass)
    polemass_length::Float32 = masspole * length  # Derived parameter
    force_mag::Float32 = 10.0f0  # Magnitude of force applied (actions scale this)
    tau::Float32 = 0.02f0  # Time step for simulation
    
    # Episode limits
    theta_threshold_radians::Float32 = 12.0f0 * 2 * π / 360  # ±12 degrees
    x_threshold::Float32 = 2.4f0  # ±2.4 units
    
    # Initial state bounds
    initial_state_bound::Float32 = 0.05f0  # All states initialized in [-0.05, 0.05]
    
    # Reward configuration
    sutton_barto_reward::Bool = false  # If true: 0 per step, -1 on termination; else +1 per step
end

mutable struct CartPoleEnv <: AbstractEnv
    problem::CartPoleProblem
    action_space::Discrete
    observation_space::Box{Float32}
    max_steps::Int
    step::Int
    rng::Random.AbstractRNG

    function CartPoleEnv(; problem=nothing, max_steps::Int=500, rng::Random.AbstractRNG=Random.Xoshiro(), kwargs...)
        # Create a problem if not provided, using kwargs for its constructor
        if isnothing(problem)
            problem = CartPoleProblem(; kwargs...)
        end

        # Discrete action space: 0 = push left, 1 = push right
        action_space = Discrete(2, 0)  # 2 actions starting from 0

        # Observation space: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
        # Use large bounds for velocities as they're theoretically unbounded
        high = Float32[
            problem.x_threshold * 2,  # Cart position bounds
            Inf,  # Cart velocity
            problem.theta_threshold_radians * 2,  # Pole angle bounds  
            Inf   # Pole angular velocity
        ]
        low = -high

        observation_space = Box{Float32}(low, high)

        env = new(problem, action_space, observation_space, max_steps, 0, rng)
        return env
    end
end

function DRiL.reset!(env::CartPoleEnv)
    reset!(env.problem, env.rng)
    env.step = 0
    nothing
end

function reset!(problem::CartPoleProblem, rng::AbstractRNG)
    # Initialize all state variables uniformly in [-initial_state_bound, initial_state_bound]
    bound = problem.initial_state_bound
    problem.x = (rand(rng, Float32) * 2 - 1) * bound
    problem.x_dot = (rand(rng, Float32) * 2 - 1) * bound  
    problem.theta = (rand(rng, Float32) * 2 - 1) * bound
    problem.theta_dot = (rand(rng, Float32) * 2 - 1) * bound
    problem.force = 0.0f0
    nothing
end

function reward(env::CartPoleEnv)
    if env.problem.sutton_barto_reward
        # Sutton-Barto style: 0 for non-terminating step, -1 for terminating step
        return terminated(env) ? -1.0f0 : 0.0f0
    else
        # Standard style: +1 for every step including termination step
        return 1.0f0
    end
end

function DRiL.act!(env::CartPoleEnv, action::AbstractArray{Int,1})
    DRiL.act!(env, action[1])
end

function DRiL.act!(env::CartPoleEnv, action::Integer)
    problem = env.problem
    
    # Convert discrete action to force: 0 -> -force_mag, 1 -> +force_mag
    force = (action == 0) ? -problem.force_mag : problem.force_mag
    problem.force = force
    
    # Cart-pole dynamics - based on Gymnasium implementation
    g = problem.gravity
    mc = problem.masscart  
    mp = problem.masspole
    mt = problem.total_mass
    l = problem.length  # Half-pole length
    mpl = problem.polemass_length
    tau = problem.tau
    
    x, x_dot, theta, theta_dot = problem.x, problem.x_dot, problem.theta, problem.theta_dot
    
    costheta = cos(theta)
    sintheta = sin(theta)
    
    # Physics equations from Gymnasium CartPole
    temp = (force + mpl * theta_dot^2 * sintheta) / mt
    thetaacc = (g * sintheta - costheta * temp) / (l * (4.0f0/3.0f0 - mp * costheta^2 / mt))
    xacc = temp - mpl * thetaacc * costheta / mt
    
    # Update state using Euler integration
    problem.x += tau * x_dot
    problem.x_dot += tau * xacc
    problem.theta += tau * theta_dot  
    problem.theta_dot += tau * thetaacc
    
    env.step += 1
    return reward(env)
end

function DRiL.observe(env::CartPoleEnv)
    p = env.problem
    return [p.x, p.x_dot, p.theta, p.theta_dot]
end

function DRiL.terminated(env::CartPoleEnv)
    p = env.problem
    # Episode terminates if pole angle or cart position exceeds bounds
    return abs(p.theta) > p.theta_threshold_radians || 
           abs(p.x) > p.x_threshold
end

DRiL.truncated(env::CartPoleEnv) = env.step >= env.max_steps
DRiL.action_space(env::CartPoleEnv) = env.action_space  
DRiL.observation_space(env::CartPoleEnv) = env.observation_space
DRiL.get_info(env::CartPoleEnv) = Dict{String, Any}("step" => env.step, "x" => env.problem.x, "theta" => env.problem.theta) 