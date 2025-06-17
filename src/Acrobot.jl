@kwdef mutable struct AcrobotProblem
    # State variables [theta1, theta2, dtheta1, dtheta2]
    theta1::Float32 = 0.0f0    # Angle of first link (radians)
    theta2::Float32 = 0.0f0    # Angle of second link relative to first (radians)
    dtheta1::Float32 = 0.0f0   # Angular velocity of first link (rad/s)
    dtheta2::Float32 = 0.0f0   # Angular velocity of second link (rad/s)
    torque::Float32 = 0.0f0    # Applied torque (action)
    
    # Physical parameters
    dt::Float32 = 0.2f0              # Integration timestep
    link_length_1::Float32 = 1.0f0   # Length of first link [m]
    link_length_2::Float32 = 1.0f0   # Length of second link [m]
    link_mass_1::Float32 = 1.0f0     # Mass of first link [kg]
    link_mass_2::Float32 = 1.0f0     # Mass of second link [kg]
    link_com_pos_1::Float32 = 0.5f0  # Center of mass position of link 1 [m]
    link_com_pos_2::Float32 = 0.5f0  # Center of mass position of link 2 [m]
    link_moi::Float32 = 1.0f0        # Moment of inertia for both links
    gravity::Float32 = 9.8f0         # Gravitational acceleration [m/s²]
    
    # Velocity bounds
    max_vel_1::Float32 = 4.0f0 * π   # Maximum angular velocity for link 1
    max_vel_2::Float32 = 9.0f0 * π   # Maximum angular velocity for link 2
    
    # Available torque values for discrete actions
    avail_torque::Vector{Float32} = [-1.0f0, 0.0f0, 1.0f0]
    
    # Dynamics variant ("book" or "nips")
    book_or_nips::String = "book"
end

mutable struct AcrobotEnv <: AbstractEnv
    problem::AcrobotProblem
    action_space::Discrete
    observation_space::Box{Float32}
    max_steps::Int
    step::Int
    rng::Random.AbstractRNG

    function AcrobotEnv(; problem=nothing, max_steps::Int=500, rng::Random.AbstractRNG=Random.Xoshiro(), kwargs...)
        # Create a problem if not provided, using kwargs for its constructor
        if isnothing(problem)
            problem = AcrobotProblem(; kwargs...)
        end

        # Discrete action space: 0 = torque -1, 1 = torque 0, 2 = torque +1
        action_space = Discrete(3, 0)

        # Observation space: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ̇1, θ̇2]
        observation_space = Box{Float32}(
            [-1.0f0, -1.0f0, -1.0f0, -1.0f0, -problem.max_vel_1, -problem.max_vel_2],
            [1.0f0, 1.0f0, 1.0f0, 1.0f0, problem.max_vel_1, problem.max_vel_2]
        )

        env = new(problem, action_space, observation_space, max_steps, 0, rng)
        return env
    end
end

function DRiL.reset!(env::AcrobotEnv)
    reset!(env.problem, env.rng)
    env.step = 0
    nothing
end

function reset!(problem::AcrobotProblem, rng::AbstractRNG; low::Float32=-0.1f0, high::Float32=0.1f0)
    # Initialize state uniformly in [low, high] for all 4 state variables
    problem.theta1 = rand(rng, Float32) * (high - low) + low
    problem.theta2 = rand(rng, Float32) * (high - low) + low  
    problem.dtheta1 = rand(rng, Float32) * (high - low) + low
    problem.dtheta2 = rand(rng, Float32) * (high - low) + low
    problem.torque = 0.0f0
    nothing
end

function reward(env::AcrobotEnv)
    # Reward is -1 for each step, 0 when goal is reached
    return terminated(env) ? 0.0f0 : -1.0f0
end

# Helper functions for angle wrapping and bounding
function wrap_angle(x::Float32, m::Float32, M::Float32)
    """Wrap angle x to be within [m, M]"""
    diff = M - m
    while x > M
        x = x - diff
    end
    while x < m
        x = x + diff
    end
    return x
end

function bound_value(x::Float32, m::Float32, M::Float32)
    """Bound value x to be within [m, M]"""
    return min(max(x, m), M)
end

# Dynamics function for RK4 integration
function acrobot_dynamics!(problem::AcrobotProblem, s_augmented::Vector{Float32})
    """Compute derivatives [dtheta1, dtheta2, ddtheta1, ddtheta2, 0] for RK4 integration"""
    m1 = problem.link_mass_1
    m2 = problem.link_mass_2
    l1 = problem.link_length_1
    lc1 = problem.link_com_pos_1
    lc2 = problem.link_com_pos_2
    I1 = problem.link_moi
    I2 = problem.link_moi
    g = problem.gravity
    a = s_augmented[5]  # Applied torque
    
    theta1 = s_augmented[1]
    theta2 = s_augmented[2]
    dtheta1 = s_augmented[3]
    dtheta2 = s_augmented[4]
    
    # Dynamics equations from Gymnasium/Sutton & Barto
    d1 = m1 * lc1^2 + m2 * (l1^2 + lc2^2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
    d2 = m2 * (lc2^2 + l1 * lc2 * cos(theta2)) + I2
    phi2 = m2 * lc2 * g * cos(theta1 + theta2 - π/2)
    phi1 = (-m2 * l1 * lc2 * dtheta2^2 * sin(theta2) 
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - π/2) 
            + phi2)
    
    if problem.book_or_nips == "nips"
        # NIPS paper version
        ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2^2 + I2 - d2^2 / d1)
    else
        # Book version (default)
        ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1^2 * sin(theta2) - phi2) / 
                   (m2 * lc2^2 + I2 - d2^2 / d1)
    end
    
    ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
    
    return [dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0f0]
end

# RK4 integration step
function rk4_step(problem::AcrobotProblem, state::Vector{Float32}, dt::Float32)
    """Single RK4 integration step"""
    k1 = acrobot_dynamics!(problem, state)
    k2 = acrobot_dynamics!(problem, state .+ (dt/2) .* k1)
    k3 = acrobot_dynamics!(problem, state .+ (dt/2) .* k2) 
    k4 = acrobot_dynamics!(problem, state .+ dt .* k3)
    
    new_state = state .+ (dt/6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
    return new_state[1:4]  # Return only the state variables, not the torque
end

function DRiL.act!(env::AcrobotEnv, action::AbstractArray{Int,1})
    DRiL.act!(env, action[1])
end

function DRiL.act!(env::AcrobotEnv, action::Integer)
    problem = env.problem
    
    # Map discrete action to torque
    torque = problem.avail_torque[action + 1]  # Convert 0-based to 1-based indexing
    problem.torque = torque
    
    # Current state
    current_state = [problem.theta1, problem.theta2, problem.dtheta1, problem.dtheta2, torque]
    
    # Integrate dynamics using RK4
    new_state = rk4_step(problem, current_state, problem.dt)
    
    # Update state with bounds and wrapping
    problem.theta1 = wrap_angle(new_state[1], -π, π)
    problem.theta2 = wrap_angle(new_state[2], -π, π)
    problem.dtheta1 = bound_value(new_state[3], -problem.max_vel_1, problem.max_vel_1)
    problem.dtheta2 = bound_value(new_state[4], -problem.max_vel_2, problem.max_vel_2)
    
    env.step += 1
    return reward(env)
end

function DRiL.observe(env::AcrobotEnv)
    p = env.problem
    # Return [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ̇1, θ̇2]
    return [cos(p.theta1), sin(p.theta1), cos(p.theta2), sin(p.theta2), p.dtheta1, p.dtheta2]
end

function DRiL.terminated(env::AcrobotEnv)
    p = env.problem
    # Goal condition: -cos(θ1) - cos(θ2 + θ1) > 1.0
    return -cos(p.theta1) - cos(p.theta2 + p.theta1) > 1.0f0
end

DRiL.truncated(env::AcrobotEnv) = env.step >= env.max_steps
DRiL.action_space(env::AcrobotEnv) = env.action_space
DRiL.observation_space(env::AcrobotEnv) = env.observation_space
DRiL.get_info(env::AcrobotEnv) = Dict{String, Any}(
    "step" => env.step, 
    "theta1" => env.problem.theta1, 
    "theta2" => env.problem.theta2,
    "dtheta1" => env.problem.dtheta1,
    "dtheta2" => env.problem.dtheta2
) 