# Helper function to process actions for plotting (handles both discrete and continuous)
function _process_actions_for_plotting(actions::AbstractArray, env::ClassicControlEnvironments.AbstractMountainCarEnv)
    # Handle case where actions is a vector of arrays (e.g., [[a1], [a2], ...])
    if !isempty(actions) && actions[1] isa AbstractArray
        flat_actions = [a[1] for a in actions]
    else
        flat_actions = actions
    end

    # Convert discrete actions to force values for visualization
    if DRiL.action_space(env) isa DRiL.Discrete
        # Discrete actions: 0 -> -1, 1 -> 0, 2 -> 1
        force_actions = [Float32(action == 0 ? -1 : (action == 1 ? 0 : 1)) for action in flat_actions]
    else
        # Continuous actions: just convert to Float32
        force_actions = [Float32(action) for action in flat_actions]
    end

    return force_actions
end

function _mountain_shape(x_range, amplitude=0.45)
    # Create the mountain/valley shape: cos(3*x) scaled
    return sin.(3.0 .* x_range) .* amplitude .+ 0.55
    # return amplitude * cos.(3.0 * x_range)
end

function _car_position(problem::MountainCarProblem)
    x = problem.position
    y = sin(3.0 * x) * 0.45 + 0.6  # Slightly above the mountain surface
    return Point2f(x, y)
end

function _force_arrow_coords(problem::MountainCarProblem)
    car_pos = _car_position(problem)
    force = problem.force

    # Arrow length scales with force magnitude
    arrow_length = 0.4 * abs(force)
    dx = arrow_length * sign(force)  # Force direction: positive = right, negative = left
    dy = 0.0

    color = :darkorange

    (; car_pos, dx, dy, color, force)
end

function ClassicControlEnvironments.plot(problem::MountainCarProblem)
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1], aspect=DataAspect())

    # Draw mountain landscape
    x_range = range(problem.min_position, problem.max_position, length=200)
    y_range = _mountain_shape(x_range)
    lines!(ax, x_range, y_range, linewidth=3, color=:brown)

    # Fill area below mountain
    band!(ax, x_range, fill(0.0, length(x_range)), y_range, color=(:brown, 0.3))

    # Draw goal flag
    goal_x = problem.goal_position
    goal_y = sin(3.0 * goal_x) * 0.45 + 0.55
    lines!(ax, [goal_x, goal_x], [goal_y, goal_y + 0.2], linewidth=4, color=:red)
    scatter!(ax, [goal_x + 0.05], [goal_y + 0.15], marker=:rect, markersize=15, color=:red)

    # Draw car
    car_pos = _car_position(problem)
    scatter!(ax, car_pos, marker=:rect, markersize=20, color=:blue)

    # Draw force arrow if there's significant force
    if abs(problem.force) > 1e-3
        arrow_data = _force_arrow_coords(problem)
        arrows2d!(ax, [arrow_data.car_pos], [Vec2f(arrow_data.dx, arrow_data.dy)],
            color=arrow_data.color, shaftwidth=0.05)
    end

    # Labels and formatting
    ax.xlabel = "Position"
    ax.ylabel = "Height"
    ax.title = "Mountain Car Environment"

    xlims!(ax, problem.min_position - 0.1, problem.max_position + 0.1)
    ylims!(ax, 0.0, 1.3)

    fig
end

function ClassicControlEnvironments.live_viz(problem::MountainCarProblem; size=(600, 400))
    position = Observable(problem.position)
    velocity = Observable(problem.velocity)
    force = Observable(problem.force)

    fig = Figure(size=size)
    ax = Axis(fig[1, 1], aspect=DataAspect())

    # Draw mountain landscape (static)
    x_range = range(problem.min_position, problem.max_position, length=200)
    y_range = _mountain_shape(x_range)
    lines!(ax, x_range, y_range, linewidth=3, color=:brown)
    band!(ax, x_range, fill(0.0, length(x_range)), y_range, color=(:brown, 0.3))

    # Draw goal flag (static)
    goal_x = problem.goal_position
    goal_y = sin(3.0 * goal_x) * 0.45 + 0.55
    lines!(ax, [goal_x, goal_x], [goal_y, goal_y + 0.2], linewidth=4, color=:red)
    scatter!(ax, [goal_x + 0.05], [goal_y + 0.15], marker=:rect, markersize=15, color=:red)

    # Draw car (dynamic)
    car_scatter = scatter!(ax, @lift(Point2f($position, sin(3.0 * $position) * 0.45 + 0.6)),
        marker=:rect, markersize=20, color=:blue)

    # Draw force arrow (dynamic)
    force_arrow = arrows2d!(ax,
        @lift([Point2f($position, sin(3.0 * $position) * 0.45 + 0.6)]),
        @lift([Vec2f(0.2 * abs($force) * sign($force), 0.0)]),
        color=:darkorange,
        shaftwidth=0.015,
        visible=@lift(abs($force) > 1e-3)
    )

    # Labels and formatting
    ax.xlabel = "Position"
    ax.ylabel = "Height"
    ax.title = "Mountain Car Environment"

    xlims!(ax, problem.min_position - 0.1, problem.max_position + 0.1)
    ylims!(ax, 0.0, 1.3)

    # Update function
    update_viz! = (problem) -> begin
        position[] = problem.position
        velocity[] = problem.velocity
        force[] = problem.force
    end

    return position, velocity, force, fig, update_viz!
end

function ClassicControlEnvironments.interactive_viz(env::ClassicControlEnvironments.AbstractMountainCarEnv)
    position = Observable(env.problem.position)
    velocity = Observable(env.problem.velocity)
    force = Observable(env.problem.force)
    rew = Observable(ClassicControlEnvironments.reward(env))
    min_rew = Observable(ClassicControlEnvironments.reward(env))
    auto_running = Observable(false)

    fig = Figure(size=(700, 600))
    ax = Axis(fig[1, 1], aspect=DataAspect())

    # Draw mountain landscape
    x_range = range(env.problem.min_position, env.problem.max_position, length=200)
    y_range = _mountain_shape(x_range)
    lines!(ax, x_range, y_range, linewidth=3, color=:brown)
    band!(ax, x_range, fill(0.0, length(x_range)), y_range, color=(:brown, 0.3))

    # Goal flag
    goal_x = env.problem.goal_position
    goal_y = sin(3.0 * goal_x) * 0.45 + 0.55
    lines!(ax, [goal_x, goal_x], [goal_y, goal_y + 0.2], linewidth=4, color=:red)
    scatter!(ax, [goal_x + 0.05], [goal_y + 0.15], marker=:rect, markersize=15, color=:red)

    # Car
    car_scatter = scatter!(ax, @lift(Point2f($position, sin(3.0 * $position) * 0.45 + 0.6)),
        marker=:rect, markersize=20, color=:blue)

    # Force arrow
    force_arrow = arrows2d!(ax,
        @lift([Point2f($position, sin(3.0 * $position) * 0.45 + 0.6)]),
        @lift([Vec2f(0.2 * abs($force) * sign($force), 0.0)]),
        color=:darkorange,
        shaftwidth=0.015,
        visible=@lift(abs($force) > 1e-3)
    )

    ax.xlabel = "Position"
    ax.ylabel = "Height"
    ax.title = "Mountain Car Environment"
    xlims!(ax, env.problem.min_position - 0.1, env.problem.max_position + 0.1)
    ylims!(ax, 0.0, 1.3)

    # Reward display
    rew_ax = Axis(fig[1, 2], title="Reward", limits=@lift((nothing, ($min_rew, 100))))
    rew_bar = barplot!(rew_ax, 1, rew)
    colsize!(fig.layout, 2, Relative(0.25))

    # Control slider - different setup for discrete vs continuous
    is_discrete = DRiL.action_space(env) isa DRiL.Discrete
    if is_discrete
        # Discrete: slider with 3 values corresponding to forces -1, 0, 1
        sg = SliderGrid(fig[2, 1],
            (label="Force", range=[-1.0, 0.0, 1.0], startvalue=0.0),
            width=Relative(0.9)
        )
    else
        # Continuous: full range slider
        sg = SliderGrid(fig[2, 1],
            (label="Force", range=-1.0:0.01:1.0, startvalue=0.0),
            width=Relative(0.9)
        )
    end
    force_slider = sg.sliders[1]

    # Control buttons
    button_grid = GridLayout(fig[3, 1])
    start_button = Button(button_grid[1, 1], label="Start Auto", tellwidth=false)
    stop_button = Button(button_grid[1, 2], label="Stop Auto", tellwidth=false)
    step_button = Button(button_grid[1, 3], label="Single Step", tellwidth=false)
    reset_button = Button(button_grid[1, 4], label="Reset", tellwidth=false)

    current_task = Ref{Union{Task,Nothing}}(nothing)

    # Helper function to convert force to action
    function force_to_action(force_val)
        if is_discrete
            # Convert force to discrete action: -1 -> 0, 0 -> 1, 1 -> 2
            return force_val == -1.0 ? 0 : (force_val == 0.0 ? 1 : 2)
        else
            return force_val
        end
    end

    # Slider updates
    on(force_slider.value) do val
        force[] = val
    end

    # Auto-running functionality
    on(start_button.clicks) do n
        if !auto_running[]
            auto_running[] = true
            start_button.label = "Running..."
            start_button.buttoncolor = :lightgreen

            current_task[] = @async begin
                try
                    while auto_running[]
                        sleep(0.05)  # ~20 FPS
                        if auto_running[]
                            action = force_to_action(force[])
                            act!(env, action)
                            position[] = env.problem.position
                            velocity[] = env.problem.velocity
                            rew[] = ClassicControlEnvironments.reward(env)
                            min_rew[] = min(min_rew[], rew[])

                            # Check if episode ended
                            if terminated(env) || truncated(env)
                                auto_running[] = false
                                start_button.label = "Episode Ended"
                                start_button.buttoncolor = :orange
                                break
                            end
                        end
                    end
                catch e
                    @warn "Auto-stepping task interrupted: $e"
                finally
                    if auto_running[]
                        auto_running[] = false
                        start_button.label = "Start Auto"
                        start_button.buttoncolor = :lightgray
                    end
                end
            end
        end
    end

    # Stop button
    on(stop_button.clicks) do n
        if auto_running[]
            auto_running[] = false
            start_button.label = "Start Auto"
            start_button.buttoncolor = :lightgray
        end
    end

    # Single step
    on(step_button.clicks) do n
        if !auto_running[]
            action = force_to_action(force[])
            act!(env, action)
            position[] = env.problem.position
            velocity[] = env.problem.velocity
            rew[] = ClassicControlEnvironments.reward(env)
            min_rew[] = min(min_rew[], rew[])
        end
    end

    # Reset button
    on(reset_button.clicks) do n
        if !auto_running[]
            reset!(env)
            position[] = env.problem.position
            velocity[] = env.problem.velocity
            force[] = 0.0
            force_slider.value[] = 0.0
            rew[] = ClassicControlEnvironments.reward(env)
            min_rew[] = rew[]
            start_button.label = "Start Auto"
            start_button.buttoncolor = :lightgray
        end
    end

    display(fig)

    return position, velocity, force, fig, sg, start_button, stop_button, step_button, reset_button
end

function ClassicControlEnvironments.plot_trajectory(env::ClassicControlEnvironments.AbstractMountainCarEnv, observations::AbstractArray, actions::AbstractArray, rewards::AbstractArray)
    fig = Figure(size=(800, 800))
    n = length(observations)

    positions = getindex.(observations, 1)
    velocities = getindex.(observations, 2)

    # Process actions (handles both discrete and continuous)
    actions = _process_actions_for_plotting(actions, env)

    # Position plot
    ax_pos = Axis(fig[1, 1], title="Position over Time")
    pos_line = scatterlines!(ax_pos, positions, label="Position")
    hlines!(ax_pos, [env.problem.goal_position], color=:red, linestyle=:dash, label="Goal")
    hlines!(ax_pos, [env.problem.min_position, env.problem.max_position], color=:gray, linestyle=:dot, label="Bounds")

    # Velocity plot
    ax_vel = Axis(fig[1, 2], title="Velocity over Time")
    vel_line = scatterlines!(ax_vel, velocities, label="Velocity")
    hlines!(ax_vel, [-env.problem.max_speed, env.problem.max_speed], color=:gray, linestyle=:dot, label="Max Speed")

    # Action plot
    ax_action = Axis(fig[2, 1], title="Actions (Force)")
    action_line = scatterlines!(ax_action, actions, label="Force")
    hlines!(ax_action, [-1.0, 1.0], color=:gray, linestyle=:dot, label="Action Bounds")

    # Reward plot (shifted to align with resulting observations)
    ax_rew = Axis(fig[2, 2], title="Rewards")
    # Pad rewards with NaN for the first observation (no preceding action)
    shifted_rewards = [NaN; rewards]
    rew_line = scatterlines!(ax_rew, shifted_rewards, label="Reward")

    # Trajectory in 2D space (position vs velocity)
    ax_traj = Axis(fig[3, 1:2], title="Trajectory (Position vs Velocity)")
    traj_line = scatterlines!(ax_traj, positions, velocities, label="Trajectory")
    scatter!(ax_traj, [positions[1]], [velocities[1]], color=:green, markersize=10, label="Start")
    scatter!(ax_traj, [positions[end]], [velocities[end]], color=:red, markersize=10, label="End")
    vlines!(ax_traj, [env.problem.goal_position], color=:red, linestyle=:dash, label="Goal Position")
    # Add position and velocity limits
    vlines!(ax_traj, [env.problem.min_position, env.problem.max_position], color=:gray, linestyle=:dot, label="Position Bounds")
    hlines!(ax_traj, [-env.problem.max_speed, env.problem.max_speed], color=:gray, linestyle=:dot, label="Velocity Bounds")
    ax_traj.xlabel = "Position"
    ax_traj.ylabel = "Velocity"
    fig[4, 1:2] = Legend(fig, ax_traj, orientation=:horizontal)

    fig
end

function ClassicControlEnvironments.plot_trajectory_interactive(env::ClassicControlEnvironments.AbstractMountainCarEnv, observations::AbstractArray, actions::AbstractArray, rewards::AbstractArray)
    # Process actions (handles both discrete and continuous)
    processed_actions = _process_actions_for_plotting(actions, env)

    num_steps = length(observations)
    if num_steps == 0
        error("Observations array cannot be empty.")
    end
    if num_steps != length(processed_actions) + 1
        error("Observations must have one more element than actions. Observations length: $(num_steps), Actions length: $(length(processed_actions))")
    end

    # Initial state for the live visualization
    initial_obs = observations[1] # This is [position, velocity]
    initial_position = initial_obs[1]
    initial_velocity = initial_obs[2]
    initial_force = processed_actions[1]

    # Create a MountainCarProblem instance for the initial visualization
    problem_for_viz = MountainCarProblem(
        position=Float32(initial_position),
        velocity=Float32(initial_velocity),
        force=Float32(initial_force),
        min_position=env.problem.min_position,
        max_position=env.problem.max_position,
        max_speed=env.problem.max_speed,
        goal_position=env.problem.goal_position,
        goal_velocity=env.problem.goal_velocity,
        gravity=env.problem.gravity
    )

    # Get the live visualization components from live_viz
    _, _, _, fig, update_viz! = live_viz(problem_for_viz; size=(600, 600))

    # Add a slider for trajectory step
    display(fig)
    sg = SliderGrid(fig[2, 1],
        (label="Step", range=1:num_steps, startvalue=1),
        (label="Playback Speed", range=0.01:0.01:0.1, startvalue=0.05)
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
        current_position = current_obs[1]
        current_velocity = current_obs[2]
        # Handle final observation (no corresponding action)
        current_force = step_idx <= length(processed_actions) ? processed_actions[step_idx] : 0.0f0

        updated_problem = MountainCarProblem(
            position=Float32(current_position),
            velocity=Float32(current_velocity),
            force=Float32(current_force),
            min_position=env.problem.min_position,
            max_position=env.problem.max_position,
            max_speed=env.problem.max_speed,
            goal_position=env.problem.goal_position,
            goal_velocity=env.problem.goal_velocity,
            gravity=env.problem.gravity
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

    return fig, trajectory_slider, start_button, stop_button, step_button, reset_button
end

function ClassicControlEnvironments.animate_trajectory_video(env::ClassicControlEnvironments.AbstractMountainCarEnv,
    observations::AbstractArray,
    actions::AbstractArray,
    output_filename::AbstractString;
    target_fps::Int=25
)
    # Process actions (handles both discrete and continuous)
    actions = _process_actions_for_plotting(actions, env)
    num_steps = length(observations)
    if num_steps == 0
        error("Observations array cannot be empty.")
    end
    if num_steps != length(actions) + 1
        error("Observations must have one more element than actions. Observations length: $(num_steps), Actions length: $(length(actions))")
    end

    # Initial state for the live visualization
    initial_obs = observations[1]
    initial_position = initial_obs[1]
    initial_velocity = initial_obs[2]
    initial_force = actions[1]

    problem_for_viz = env.problem
    problem_for_viz.position = initial_position
    problem_for_viz.velocity = initial_velocity
    problem_for_viz.force = initial_force

    _, _, _, fig, update_viz! = live_viz(problem_for_viz)

    # Animation function
    function frame_update(step_idx)
        current_obs = observations[step_idx]
        current_position = current_obs[1]
        current_velocity = current_obs[2]
        # Handle final observation (no corresponding action)
        current_force = step_idx <= length(actions) ? actions[step_idx] : 0.0f0

        updated_problem = MountainCarProblem(
            position=Float32(current_position),
            velocity=Float32(current_velocity),
            force=Float32(current_force),
            min_position=env.problem.min_position,
            max_position=env.problem.max_position,
            max_speed=env.problem.max_speed,
            goal_position=env.problem.goal_position,
            goal_velocity=env.problem.goal_velocity,
            gravity=env.problem.gravity
        )
        update_viz!(updated_problem)
    end

    # Use dt=1 for MountainCar (no dt field) to set frame dropping for real-time video
    dt = 1.0  # MountainCar uses dt=1 implicitly
    steps_per_frame = max(1, round(Int, 1 / (target_fps * dt)))
    frame_indices = 1:steps_per_frame:num_steps

    Makie.record(fig, output_filename, frame_indices; framerate=target_fps) do step_idx
        frame_update(step_idx)
    end

    return output_filename
end

function ClassicControlEnvironments.plot_trajectory_phase_space(env::ClassicControlEnvironments.AbstractMountainCarEnv, observations::AbstractArray, actions::AbstractArray; size=(600, 400))
    positions = getindex.(observations, 1)
    velocities = getindex.(observations, 2)

    fig = Figure(size=size)
    ax = Axis(fig[1, 1], title="Trajectory (Position vs Velocity)")

    # Main trajectory line
    traj_line = scatterlines!(ax, positions, velocities, label="Trajectory")

    # Start and end points
    scatter!(ax, [positions[1]], [velocities[1]], color=:green, markersize=10, label="Start")
    scatter!(ax, [positions[end]], [velocities[end]], color=:red, markersize=10, label="End")

    # Goal position line
    vlines!(ax, [env.problem.goal_position], color=:red, linestyle=:dash, label="Goal Position")

    # Position and velocity limits
    vlines!(ax, [env.problem.min_position, env.problem.max_position], color=:gray, linestyle=:dot, label="Position Bounds")
    hlines!(ax, [-env.problem.max_speed, env.problem.max_speed], color=:gray, linestyle=:dot, label="Velocity Bounds")

    ax.xlabel = "Position"
    ax.ylabel = "Velocity"

    # Add legend below the plot
    fig[2, 1] = Legend(fig, ax, orientation=:horizontal, tellheight=true)

    return fig
end