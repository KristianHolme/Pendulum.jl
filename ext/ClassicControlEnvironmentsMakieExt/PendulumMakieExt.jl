function _pendulum_coords(L, θ)
    return Point2f(-L * sin(θ), L * cos(θ))
end

function _torque_arc_coords(L, θ, τ)
    # Arc is centered at the pivot point (origin)
    center = Point2f(0.0f0, 0.0f0)

    # Arc radius - smaller for pendulum, scaled by length
    arc_radius = 0.3f0 * L

    # Arc span based on torque magnitude
    arc_span = π / 4 + π / 8 * clamp(abs(τ) / 2, 0, 1)  # Between π/4 and 3π/8 radians

    # Start the arc from the current pendulum angle, going in torque direction
    if τ > 0  # Counter-clockwise (positive torque)
        start_angle = θ - arc_span / 2
        stop_angle = θ + arc_span / 2
        arrow_angle = stop_angle + π / 2  # Tangent to arc (counter-clockwise)
    else  # Clockwise (negative torque)
        start_angle = θ + arc_span / 2
        stop_angle = θ - arc_span / 2
        arrow_angle = stop_angle - π / 2  # Tangent to arc (clockwise)
    end

    # Arrow head position at the end of the arc
    arrow_head_pos = Point2f(
        arc_radius * sin(stop_angle),
        arc_radius * -cos(stop_angle)
    )

    # Arrow head direction vector (larger for visibility)
    head_length = 0.25f0 * L
    dx = head_length * sin(arrow_angle)
    dy = head_length * -cos(arrow_angle)

    # Enhanced color scheme
    color = :forestgreen

    # Dynamic line width
    linewidth = 2 + 2 * clamp(abs(τ) / 2, 0, 1)
    shaftwidth = 0.01 + 0.02 * clamp(abs(τ) / 2, 0, 1)

    (; center, radius=arc_radius, start_angle, stop_angle, arrow_pos=arrow_head_pos,
        dx, dy, color, linewidth, shaftwidth)
end

function ClassicControlEnvironments.plot(problem::PendulumProblem)
    L = problem.length
    θ = problem.theta
    τ = problem.torque
    fig = Figure(size=(400, 400))
    ax = Axis(fig[1, 1], aspect=1)
    # Pendulum
    pt = _pendulum_coords(L, θ)
    lines!(ax, [Point2f(0.0, 0.0), pt], linewidth=4, color=:black)
    scatter!(ax, [0.0], [0.0], color=:red, markersize=15)
    scatter!(ax, pt, color=:blue, markersize=20)
    # Torque arc and arrow - show for any non-zero torque
    if abs(τ) > 0
        arr = _torque_arc_coords(L, θ, τ)
        # Draw the arc
        arc!(ax, arr.center, arr.radius, arr.start_angle, arr.stop_angle,
            color=arr.color, linewidth=arr.linewidth)
        # Draw the arrow head
        arrows2d!(ax, [arr.arrow_pos], [Vec2f(arr.dx, arr.dy)],
            color=arr.color, shaftwidth=arr.shaftwidth, tiplength=12, tipwidth=12)
    end
    xlims!(ax, -L - 0.2, L + 0.2)
    ylims!(ax, -L - 0.2, L + 0.2)
    fig
end

function ClassicControlEnvironments.live_viz(problem::PendulumProblem)
    θ = Observable(problem.theta)
    τ = Observable(problem.torque)
    L = problem.length
    fig = Figure(size=(400, 400))
    ax = Axis(fig[1, 1], aspect=1)
    pendulum_line = lines!(ax, @lift([Point2f(0.0, 0.0), _pendulum_coords(L, $θ)]), linewidth=4, color=:black)
    scatter!(ax, [0.0], [0.0], color=:red, markersize=15)
    mass_scatter = scatter!(ax, @lift(_pendulum_coords(L, $θ)), color=:blue, markersize=20)
    # Torque arc and arrow
    torque_arc = arc!(ax,
        lift((θ, τ) -> _torque_arc_coords(L, θ, τ).center, θ, τ),
        lift((θ, τ) -> _torque_arc_coords(L, θ, τ).radius, θ, τ),
        lift((θ, τ) -> _torque_arc_coords(L, θ, τ).start_angle, θ, τ),
        lift((θ, τ) -> _torque_arc_coords(L, θ, τ).stop_angle, θ, τ),
        color=lift((θ, τ) -> _torque_arc_coords(L, θ, τ).color, θ, τ),
        linewidth=lift((θ, τ) -> _torque_arc_coords(L, θ, τ).linewidth, θ, τ),
        visible=lift((θ, τ) -> abs(τ) > 1e-3, θ, τ))

    torque_arrow = arrows2d!(ax,
        lift((θ, τ) -> [_torque_arc_coords(L, θ, τ).arrow_pos], θ, τ),
        lift((θ, τ) -> [Vec2f(_torque_arc_coords(L, θ, τ).dx, _torque_arc_coords(L, θ, τ).dy)], θ, τ),
        color=lift((θ, τ) -> _torque_arc_coords(L, θ, τ).color, θ, τ),
        shaftwidth=lift((θ, τ) -> _torque_arc_coords(L, θ, τ).shaftwidth, θ, τ),
        tiplength=12,
        tipwidth=12,
        visible=lift((θ, τ) -> abs(τ) > 1e-3, θ, τ))
    xlims!(ax, -L - 0.2, L + 0.2)
    ylims!(ax, -L - 0.2, L + 0.2)
    # display(fig)
    update_viz! = (problem) -> begin
        θ[] = problem.theta
        τ[] = problem.torque
    end
    return θ, τ, fig, update_viz!
end

function ClassicControlEnvironments.interactive_viz(env::PendulumEnv)
    θ = Observable(env.problem.theta)
    τ = Observable(env.problem.torque)
    dt = Observable(env.problem.dt)
    rew = Observable(Pendulum.reward(env))
    min_rew = Observable(Pendulum.reward(env))
    live = Observable(true)
    L = env.problem.length

    fig = Figure(size=(500, 600))
    ax = Axis(fig[1, 1], aspect=1)
    pendulum_line = lines!(ax, @lift([Point2f(0.0, 0.0), _pendulum_coords(L, $θ)]), linewidth=4, color=:black)
    scatter!(ax, [0.0], [0.0], color=:red, markersize=15)
    mass_scatter = scatter!(ax, @lift(_pendulum_coords(L, $θ)), color=:blue, markersize=20)
    torque_arc = arc!(ax,
        lift((θ, τ) -> _torque_arc_coords(L, θ, τ).center, θ, τ),
        lift((θ, τ) -> _torque_arc_coords(L, θ, τ).radius, θ, τ),
        lift((θ, τ) -> _torque_arc_coords(L, θ, τ).start_angle, θ, τ),
        lift((θ, τ) -> _torque_arc_coords(L, θ, τ).stop_angle, θ, τ),
        color=lift((θ, τ) -> _torque_arc_coords(L, θ, τ).color, θ, τ),
        linewidth=lift((θ, τ) -> _torque_arc_coords(L, θ, τ).linewidth, θ, τ),
        visible=lift((θ, τ) -> abs(τ) > 1e-3, θ, τ))

    torque_arrow = arrows2d!(ax,
        lift((θ, τ) -> [_torque_arc_coords(L, θ, τ).arrow_pos], θ, τ),
        lift((θ, τ) -> [Vec2f(_torque_arc_coords(L, θ, τ).dx, _torque_arc_coords(L, θ, τ).dy)], θ, τ),
        color=lift((θ, τ) -> _torque_arc_coords(L, θ, τ).color, θ, τ),
        shaftwidth=lift((θ, τ) -> _torque_arc_coords(L, θ, τ).shaftwidth, θ, τ),
        tiplength=12,
        tipwidth=12,
        visible=lift((θ, τ) -> abs(τ) > 1e-3, θ, τ))
    xlims!(ax, -L - 0.2, L + 0.2)
    ylims!(ax, -L - 0.2, L + 0.2)

    rew_ax = Axis(fig[1, 2], title="Reward", limits=@lift((nothing, ($min_rew, 0))))
    rew_bar = barplot!(rew_ax, 1, rew)
    colsize!(fig.layout, 2, Relative(0.3))

    # SliderGrid for torque and dt
    sg = SliderGrid(fig[2, 1],
        (label="Torque", range=-2.0:0.01:2.0, startvalue=env.problem.torque),
        (label="dt", range=0.0001:0.0001:0.01, startvalue=env.problem.dt),
        width=Relative(0.9)
    )

    # Control buttons for automatic stepping
    button_grid = GridLayout(fig[3, 1])
    start_button = Button(button_grid[1, 1], label="Start Auto", tellwidth=false)
    stop_button = Button(button_grid[1, 2], label="Stop Auto", tellwidth=false)
    step_button = Button(button_grid[1, 3], label="Single Step", tellwidth=false)

    # Button states
    auto_running = Observable(false)
    current_task = Ref{Union{Task,Nothing}}(nothing)

    torque_slider = sg.sliders[1]
    dt_slider = sg.sliders[2]
    on(torque_slider.value) do val
        τ[] = val
    end
    on(dt_slider.value) do val
        dt[] = val
        env.problem.dt = val
    end

    # Start button functionality
    on(start_button.clicks) do n
        if !auto_running[]
            auto_running[] = true
            start_button.label = "Running..."
            start_button.buttoncolor = :lightgreen

            # Start the automatic stepping task
            current_task[] = @async begin
                try
                    while auto_running[]
                        sleep(dt[])
                        if auto_running[]  # Check again after sleep
                            act!(env, τ[])
                            θ[] = env.problem.theta
                            rew[] = Pendulum.reward(env)
                            min_rew[] = min(min_rew[], rew[])
                        end
                    end
                catch e
                    @warn "Auto-stepping task interrupted: $e"
                finally
                    auto_running[] = false
                    start_button.label = "Start Auto"
                    start_button.buttoncolor = :lightgray
                end
            end
        end
    end

    # Stop button functionality
    on(stop_button.clicks) do n
        if auto_running[]
            auto_running[] = false
            start_button.label = "Start Auto"
            start_button.buttoncolor = :lightgray
            if !isnothing(current_task[])
                # Give the task a moment to finish cleanly
                sleep(0.01)
            end
        end
    end

    # Single step button functionality
    on(step_button.clicks) do n
        if !auto_running[]  # Only allow single steps when not auto-running
            act!(env, τ[])
            θ[] = env.problem.theta
            rew[] = reward(env)
            min_rew[] = min(min_rew[], rew[])
        end
    end

    display(fig)

    return θ, τ, dt, fig, sg, start_button, stop_button, step_button
end

function ClassicControlEnvironments.plot_trajectory(env::PendulumEnv, observations::AbstractArray, actions::AbstractArray, rewards::AbstractArray)
    fig = Figure(size=(800, 600))
    n = length(observations)
    xs = getindex.(observations, 1)
    ys = getindex.(observations, 2)
    vels = getindex.(observations, 3)
    scaled_vels = vels .* 8
    thetas = atan.(ys, xs)

    actions = vec(stack(actions))
    torques = actions .* 2

    individual_rewards = Pendulum.pendulum_rewards.(thetas[2:end], scaled_vels[2:end], torques[1:end-1])
    theta_rewards = getindex.(individual_rewards, 1)
    vel_rewards = getindex.(individual_rewards, 2)
    torque_rewards = getindex.(individual_rewards, 3)

    ax_angle = Axis(fig[1, 1], title="Angle", limits=((nothing), (-π, π)))
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
    # Shift rewards to align with resulting observations
    shifted_total_rewards = [NaN; rewards]
    shifted_theta_rewards = [NaN; theta_rewards]
    shifted_vel_rewards = [NaN; vel_rewards]
    shifted_torque_rewards = [NaN; torque_rewards]
    rew_plot = scatterlines!(ax_rew, shifted_total_rewards, label="Total")
    theta_rew_plot = scatterlines!(ax_rew, shifted_theta_rewards, label="Theta")
    vel_rew_plot = scatterlines!(ax_rew, shifted_vel_rewards, label="Velocity")
    torque_rew_plot = scatterlines!(ax_rew, shifted_torque_rewards, label="Torque")
    axislegend(ax_rew, location=:rb)

    fig
end

function ClassicControlEnvironments.plot_trajectory_interactive(env::PendulumEnv, observations::AbstractArray, actions::AbstractArray, rewards::AbstractArray)
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
    if num_steps != length(processed_actions_scaled) + 1
        error("Observations must have one more element than actions. Observations length: $(num_steps), Actions length: $(length(processed_actions_scaled))")
    end

    # Initial state for the live visualization
    initial_obs = observations[1] # This is [cos(theta), sin(theta), scaled_vel]
    initial_theta = atan(initial_obs[2], initial_obs[1]) # obs[2] is sin(theta), obs[1] is cos(theta)
    initial_torque_scaled = processed_actions_scaled[1]

    # Create a PendulumProblem instance for the initial visualization
    # Velocity and dt from env.problem are used, or could be set to defaults if not relevant for viz
    problem_for_viz = PendulumProblem(
        theta=Float32(initial_theta),
        velocity=0.0f0, # Not directly used by live_viz for display logic
        torque=Float32(initial_torque_scaled),
        mass=env.problem.mass,
        length=env.problem.length,
        gravity=env.problem.gravity,
        dt=env.problem.dt # Also not directly used by display but part of struct
    )

    # Get the live visualization components from live_viz
    # live_viz returns: θ_observable, τ_observable, fig, update_viz_function
    # We don't need the observables here as update_viz! handles them.
    _, _, fig, update_viz! = live_viz(problem_for_viz)

    # Adjust figure layout slightly if needed, e.g., increase height for the slider
    # fig.size = (400, 450) # Example: uncomment and adjust if slider feels cramped

    # Add a slider for trajectory step
    # live_viz uses fig[1,1] for its Axis. We add the slider in a new row.
    # Ensure there's a layout cell available or create one.
    # By default, fig from live_viz is a 1x1 grid for the axis.
    # We can add to fig[2,1]. Makie should handle expanding the layout.
    display(fig)
    sg = SliderGrid(fig[2, 1],
        (label="Step", range=1:num_steps, startvalue=1),
        (label="Playback Speed", range=0.01:0.01:0.5, startvalue=0.05)
    )
    trajectory_slider = sg.sliders[1]
    speed_slider = sg.sliders[2]

    # Control buttons for automatic trajectory playback
    button_grid = GridLayout(fig[3, 1])
    start_button = Button(button_grid[1, 1], label="Play", tellwidth=false)
    stop_button = Button(button_grid[1, 2], label="Pause", tellwidth=false)
    step_button = Button(button_grid[1, 3], label="Next Step", tellwidth=false)
    reset_button = Button(button_grid[1, 4], label="Reset", tellwidth=false)

    # Button states
    auto_playing = Observable(false)
    current_task = Ref{Union{Task,Nothing}}(nothing)

    # Function to update visualization for a given step
    function update_step!(step_idx)
        current_obs = observations[step_idx]
        current_theta = atan(current_obs[2], current_obs[1])
        # Handle final observation (no corresponding action)
        current_torque_val_scaled = step_idx <= length(processed_actions_scaled) ? processed_actions_scaled[step_idx] : 0.0f0

        updated_problem = PendulumProblem(
            theta=Float32(current_theta),
            velocity=0.0f0, # As per requirement, velocity from trajectory not used here
            torque=Float32(current_torque_val_scaled),
            mass=env.problem.mass,
            length=env.problem.length,
            gravity=env.problem.gravity,
            dt=env.problem.dt
        )
        update_viz!(updated_problem)
    end

    # Manual slider control
    on(trajectory_slider.value) do step_idx
        if !auto_playing[]  # Only respond to manual slider changes when not auto-playing
            update_step!(step_idx)
        end
    end

    # Start/Play button functionality
    on(start_button.clicks) do n
        if !auto_playing[]
            auto_playing[] = true
            start_button.label = "Playing..."
            start_button.buttoncolor = :lightgreen

            # Start the automatic playback task
            current_task[] = @async begin
                try
                    current_step = trajectory_slider.value[]
                    while auto_playing[] && current_step <= num_steps
                        sleep(speed_slider.value[])
                        if auto_playing[]  # Check again after sleep
                            # Update slider position and visualization
                            set_close_to!(trajectory_slider, current_step)
                            update_step!(current_step)
                            current_step += 1

                            # Stop at end of trajectory
                            if current_step > num_steps
                                auto_playing[] = false
                                break
                            end
                        end
                    end
                catch e
                    @warn "Auto-playback task interrupted: $e"
                finally
                    auto_playing[] = false
                    start_button.label = "Play"
                    start_button.buttoncolor = :lightgray
                end
            end
        end
    end

    # Stop/Pause button functionality
    on(stop_button.clicks) do n
        if auto_playing[]
            auto_playing[] = false
            start_button.label = "Play"
            start_button.buttoncolor = :lightgray
            if !isnothing(current_task[])
                # Give the task a moment to finish cleanly
                sleep(0.01)
            end
        end
    end

    # Single step button functionality
    on(step_button.clicks) do n
        if !auto_playing[]  # Only allow single steps when not auto-playing
            current_step = min(trajectory_slider.value[] + 1, num_steps)
            trajectory_slider.value[] = current_step
            notify(trajectory_slider)
            update_step!(current_step)
        end
    end

    # Reset button functionality
    on(reset_button.clicks) do n
        if !auto_playing[]  # Only allow reset when not auto-playing
            trajectory_slider.value[] = 1
            notify(trajectory_slider)
            update_step!(1)
        end
    end

    # live_viz already calls display(fig), so no need to call it again here.
    return fig, trajectory_slider, start_button, stop_button, step_button, reset_button
end

function ClassicControlEnvironments.animate_trajectory_video(env::PendulumEnv,
    observations::AbstractArray,
    actions::AbstractArray,
    output_filename::AbstractString;
    target_fps::Int=25
)
    # Use actions directly (assume already scaled)
    if actions[1] isa AbstractArray
        actions = first.(actions)
    end
    num_steps = length(observations)
    if num_steps == 0
        error("Observations array cannot be empty.")
    end
    if num_steps != length(actions) + 1
        error("Observations must have one more element than actions. Observations length: $(num_steps), Actions length: $(length(actions))")
    end
    # Initial state for the live visualization
    initial_obs = observations[1]
    initial_theta = atan(initial_obs[2], initial_obs[1])
    initial_torque = actions[1]
    problem_for_viz = env.problem
    problem_for_viz.theta = initial_theta
    problem_for_viz.torque = initial_torque
    problem_for_viz.velocity = 0.0f0

    _, _, fig, update_viz! = live_viz(problem_for_viz)
    # Animation function
    function frame_update(step_idx)
        current_obs = observations[step_idx]
        current_theta = atan(current_obs[2], current_obs[1])
        # Handle final observation (no corresponding action)
        current_torque = step_idx <= length(actions) ? actions[step_idx] : 0.0f0
        updated_problem = PendulumProblem(
            theta=Float32(current_theta),
            velocity=0.0f0,
            torque=Float32(current_torque),
            mass=env.problem.mass,
            length=env.problem.length,
            gravity=env.problem.gravity,
            dt=env.problem.dt
        )
        update_viz!(updated_problem)
    end
    # Use dt from env.problem to set frame dropping for real-time video
    dt = env.problem.dt
    steps_per_frame = max(1, round(Int, 1 / (target_fps * dt)))
    frame_indices = 1:steps_per_frame:num_steps
    Makie.record(fig, output_filename, frame_indices; framerate=target_fps) do step_idx
        frame_update(step_idx)
    end
    return output_filename
end