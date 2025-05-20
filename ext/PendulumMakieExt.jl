module PendulumMakieExt

using Makie
using Pendulum

function _pendulum_coords(L, θ)
    return Point2f(-L * sin(θ), L * cos(θ))
end

function _torque_arrow_coords(L, θ, τ)
    # Arrow is centered at the midpoint of the pendulum
    mid_x, mid_y = _pendulum_coords(L/2, θ)
    # Arrow direction: perpendicular to pendulum
    perp_angle = θ + π/2 * sign(τ)
    # Arrow length scales with torque, now up to 0.8*L
    arrow_length = 0.8 * L * clamp(abs(τ) / 2, 0, 1)
    dx = arrow_length * -sin(perp_angle)
    dy = arrow_length * cos(perp_angle)
    color = τ > 0 ? :green : :orange
    (; mid_x, mid_y, dx, dy, color)
end

function plot_pendulum(problem::PendulumProblem)
    L = problem.length
    θ = problem.theta
    τ = problem.torque
    fig = Figure(size = (400, 400))
    ax = Axis(fig[1, 1], aspect = 1)
    # Pendulum
    pt = _pendulum_coords(L, θ)
    lines!(ax, [Point2f(0.0, 0.0), pt], linewidth=4, color=:black)
    scatter!(ax, [0.0], [0.0], color=:red, markersize=15)
    scatter!(ax, pt, color=:blue, markersize=20)
    # Torque arrow
    if abs(τ) > 1e-4
        arr = _torque_arrow_coords(L, θ, τ)
        arrows!(ax, [Point2f(arr.mid_x, arr.mid_y)], [Vec2f(arr.dx, arr.dy)], color=arr.color, arrowsize=0.2)
    end
    xlims!(ax, -L-0.2, L+0.2)
    ylims!(ax, -L-0.2, L+0.2)
    fig
end

function live_pendulum_viz(problem::PendulumProblem)
    θ = Observable(problem.theta)
    τ = Observable(problem.torque)
    L = problem.length
    fig = Figure(size = (400, 400))
    ax = Axis(fig[1, 1], aspect = 1)
    pendulum_line = lines!(ax, @lift([Point2f(0.0, 0.0), _pendulum_coords(L, $θ)]), linewidth=4, color=:black)
    scatter!(ax, [0.0], [0.0], color=:red, markersize=15)
    mass_scatter = scatter!(ax, @lift(_pendulum_coords(L, $θ)), color=:blue, markersize=20)
    # Torque arrow
    torque_arrow = arrows!(ax,
        lift((θ, τ) -> [Point2f(_torque_arrow_coords(L, θ, τ).mid_x, _torque_arrow_coords(L, θ, τ).mid_y)], θ, τ),
        lift((θ, τ) -> [Vec2f(_torque_arrow_coords(L, θ, τ).dx, _torque_arrow_coords(L, θ, τ).dy)], θ, τ),
        color=lift((θ, τ) -> _torque_arrow_coords(L, θ, τ).color, θ, τ), arrowsize=0.2)
    xlims!(ax, -L-0.2, L+0.2)
    ylims!(ax, -L-0.2, L+0.2)
    display(fig)
    update_viz! = (problem) -> begin
        θ[] = problem.theta
        τ[] = problem.torque
    end
    return θ, τ, fig, update_viz!
end

function interactive_viz(env::PendulumEnv)
    θ = Observable(env.problem.theta)
    τ = Observable(env.problem.torque)
    dt = Observable(env.problem.dt)
    rew = Observable(reward(env))
    min_rew = Observable(reward(env))
    live = Observable(true)
    L = env.problem.length

    fig = Figure(size = (500, 600))
    ax = Axis(fig[1, 1], aspect = 1)
    pendulum_line = lines!(ax, @lift([Point2f(0.0, 0.0), _pendulum_coords(L, $θ)]), linewidth=4, color=:black)
    scatter!(ax, [0.0], [0.0], color=:red, markersize=15)
    mass_scatter = scatter!(ax, @lift(_pendulum_coords(L, $θ)), color=:blue, markersize=20)
    torque_arrow = arrows!(ax,
        lift((θ, τ) -> [Point2f(_torque_arrow_coords(L, θ, τ).mid_x, _torque_arrow_coords(L, θ, τ).mid_y)], θ, τ),
        lift((θ, τ) -> [Vec2f(_torque_arrow_coords(L, θ, τ).dx, _torque_arrow_coords(L, θ, τ).dy)], θ, τ),
        color=lift((θ, τ) -> _torque_arrow_coords(L, θ, τ).color, θ, τ), arrowsize=0.2)
    xlims!(ax, -L-0.2, L+0.2)
    ylims!(ax, -L-0.2, L+0.2)

    rew_ax = Axis(fig[1,2], title="Reward", limits=@lift((nothing, ($min_rew, 0))))
    rew_bar = barplot!(rew_ax, 1, rew)
    colsize!(fig.layout, 2, Relative(0.3))

    # SliderGrid for torque and dt
    sg = SliderGrid(fig[2, 1],
        (label = "Torque", range = -2.0:0.01:2.0, startvalue = env.problem.torque),
        (label = "dt", range = 0.0001:0.0001:0.01, startvalue = env.problem.dt),
        width = Relative(0.9)
    )
    live_button = Button(fig[3, 1], label = "Stop", tellwidth = false)
    on(live_button.clicks) do n
        live[] = !live[]
    end
    torque_slider = sg.sliders[1]
    dt_slider = sg.sliders[2]
    on(torque_slider.value) do val
        τ[] = val
    end
    on(dt_slider.value) do val
        dt[] = val
        env.problem.dt = val
    end

    display(fig)

    @async begin
        while live[]
            sleep(dt[])
            act!(env, τ[])
            θ[] = env.problem.theta
            rew[] = reward(env)
            min_rew[] = min(min_rew[], rew[])
        end
    end

    return θ, τ, dt, fig, sg
end

function plot_trajectory(env::PendulumEnv, observations::AbstractArray, actions::AbstractArray, rewards::AbstractArray)
    fig = Figure(size=(800, 600))
    n = length(observations)
    xs = getindex.(observations, 1)
    ys = getindex.(observations, 2)
    vels = getindex.(observations, 3)
    scaled_vels = vels .* 8
    thetas = angle.(xs, ys)

    actions = vec(stack(actions))
    torques = actions .* 2

    individual_rewards = pendulum_rewards.(thetas[2:end], scaled_vels[2:end], torques[1:end-1])
    theta_rewards = getindex.(individual_rewards, 1)
    vel_rewards = getindex.(individual_rewards, 2)
    torque_rewards = getindex.(individual_rewards, 3)

    ax_angle = Axis(fig[1, 1], title="Angle", limits=((nothing),(-π, π)))
    angle_plot = scatterlines!(ax_angle, thetas)
    ax_xy = Axis(fig[1, 2], title="XY")
    x_plot = scatterlines!(ax_xy, xs, label="x")
    y_plot = scatterlines!(ax_xy, ys, label="y")
    axislegend(ax_xy)
    ax_vel = Axis(fig[2, 1], title="Velocity")
    vel_plot = scatterlines!(ax_vel, vels)

    ax_action = Axis(fig[2, 2], title="Action")
    action_plot = scatterlines!(ax_action, actions)

    ax_rew = Axis(fig[3, 1], title="Reward")
    rew_plot = scatterlines!(ax_rew, rewards, label="Total")
    theta_rew_plot = scatterlines!(ax_rew, theta_rewards, label="Theta")
    vel_rew_plot = scatterlines!(ax_rew, vel_rewards, label="Velocity")
    torque_rew_plot = scatterlines!(ax_rew, torque_rewards, label="Torque")
    axislegend(ax_rew)

    fig
end

function plot_trajectory_interactive(env::PendulumEnv, observations::AbstractArray, actions::AbstractArray, rewards::AbstractArray)
    # Process actions: scale by 2 and ensure they are a flat Vector{Float32}
    local processed_actions_scaled::Vector{Float32}
    if !isempty(actions) && actions[1] isa AbstractArray
        # Assuming actions is a vector of 1-element vectors e.g. [[a1], [a2], ...]
        processed_actions_scaled = [Float32(a[1] * 2.0f0) for a in actions]
    else
        # Assuming actions is already a vector of scalars e.g. [a1, a2, ...]
        processed_actions_scaled = [Float32(a * 2.0f0) for a in actions]
    end

    num_steps = length(observations)
    if num_steps == 0
        error("Observations array cannot be empty.")
    end
    if num_steps != length(processed_actions_scaled)
        error("Observations and processed actions must have the same length. Original actions length: $(length(actions)), Processed actions length: $(length(processed_actions_scaled))")
    end

    # Initial state for the live visualization
    initial_obs = observations[1] # This is [cos(theta), sin(theta), scaled_vel]
    initial_theta = atan(initial_obs[2], initial_obs[1]) # obs[2] is sin(theta), obs[1] is cos(theta)
    initial_torque_scaled = processed_actions_scaled[1]

    # Create a PendulumProblem instance for the initial visualization
    # Velocity and dt from env.problem are used, or could be set to defaults if not relevant for viz
    problem_for_viz = PendulumProblem(
        theta    = Float32(initial_theta),
        velocity = 0.0f0, # Not directly used by live_pendulum_viz for display logic
        torque   = Float32(initial_torque_scaled),
        mass     = env.problem.mass,
        length   = env.problem.length,
        gravity  = env.problem.gravity,
        dt       = env.problem.dt # Also not directly used by display but part of struct
    )

    # Get the live visualization components from live_pendulum_viz
    # live_pendulum_viz returns: θ_observable, τ_observable, fig, update_viz_function
    # We don't need the observables here as update_viz! handles them.
    _, _, fig, update_viz! = live_pendulum_viz(problem_for_viz)

    # Adjust figure layout slightly if needed, e.g., increase height for the slider
    # fig.size = (400, 450) # Example: uncomment and adjust if slider feels cramped

    # Add a slider for trajectory step
    # live_pendulum_viz uses fig[1,1] for its Axis. We add the slider in a new row.
    # Ensure there's a layout cell available or create one.
    # By default, fig from live_pendulum_viz is a 1x1 grid for the axis.
    # We can add to fig[2,1]. Makie should handle expanding the layout.
    sg = SliderGrid(fig[2,1],
        (label="Step", range=1:num_steps, startvalue=1)
    )
    trajectory_slider = sg.sliders[1]
    
    # Label(fig[2, 1, Top()], "Trajectory Step", valign = :bottom, padding = (0, 0, 5, 0)) # Alternative way to add label

    on(trajectory_slider.value) do step_idx
        current_obs = observations[step_idx]
        current_theta = atan(current_obs[2], current_obs[1])
        current_torque_val_scaled = processed_actions_scaled[step_idx]

        updated_problem = PendulumProblem(
            theta    = Float32(current_theta),
            velocity = 0.0f0, # As per requirement, velocity from trajectory not used here
            torque   = Float32(current_torque_val_scaled),
            mass     = env.problem.mass,
            length   = env.problem.length,
            gravity  = env.problem.gravity,
            dt       = env.problem.dt
        )
        update_viz!(updated_problem)
    end

    # live_pendulum_viz already calls display(fig), so no need to call it again here.
    return fig, trajectory_slider
end

export plot_pendulum, live_pendulum_viz, interactive_viz, plot_trajectory, plot_trajectory_interactive

end