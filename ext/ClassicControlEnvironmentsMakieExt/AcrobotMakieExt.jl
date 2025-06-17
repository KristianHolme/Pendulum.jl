# Helper functions for Acrobot visualization

function _link_positions(problem::AcrobotProblem)
    """Calculate the positions of both links and joints"""
    l1 = problem.link_length_1
    l2 = problem.link_length_2
    θ1 = problem.theta1
    θ2 = problem.theta2
    
    # Base joint (fixed)
    joint1 = Point2f(0.0f0, 0.0f0)
    
    # End of first link (second joint)
    x1 = -l1 * cos(θ1)
    y1 = l1 * sin(θ1)
    joint2 = Point2f(x1, y1)
    
    # End of second link (tip)
    x2 = x1 - l2 * cos(θ1 + θ2)
    y2 = y1 + l2 * sin(θ1 + θ2)
    tip = Point2f(x2, y2)
    
    return joint1, joint2, tip
end

function _goal_height_line(problem::AcrobotProblem)
    """Calculate the goal height line positions"""
    # Goal condition: -cos(θ1) - cos(θ2 + θ1) > 1.0
    # This corresponds to tip being above height 1.0
    goal_height = 1.0f0
    
    # Line extends across the visualization area
    total_length = problem.link_length_1 + problem.link_length_2
    margin = 0.5f0
    x_left = -(total_length + margin)
    x_right = total_length + margin
    
    return [Point2f(x_left, goal_height), Point2f(x_right, goal_height)]
end

function _torque_arrow_coords(problem::AcrobotProblem)
    """Calculate torque arrow position and direction"""
    joint1, joint2, _ = _link_positions(problem)
    
    torque = problem.torque
    if abs(torque) < 1e-3
        return nothing
    end
    
    # Arrow length scales with torque magnitude
    arrow_length = 0.3f0 * abs(torque)
    
    # Torque direction (positive is counter-clockwise)
    angle_offset = torque > 0 ? π/2 : -π/2
    dx = arrow_length * cos(angle_offset)
    dy = arrow_length * sin(angle_offset)
    
    color = torque > 0 ? :green : :red
    
    return (; pos=joint2, dx=dx, dy=dy, color=color, torque=torque)
end

function ClassicControlEnvironments.plot(problem::AcrobotProblem)
    fig = Figure(size=(600, 600))
    ax = Axis(fig[1, 1], aspect=DataAspect())
    
    # Calculate link positions
    joint1, joint2, tip = _link_positions(problem)
    
    # Draw the links
    lines!(ax, [joint1, joint2], linewidth=8, color=:blue, label="Link 1")
    lines!(ax, [joint2, tip], linewidth=8, color=:blue, label="Link 2")
    
    # Draw joints
    scatter!(ax, [joint1], markersize=15, color=:black, label="Base Joint")
    scatter!(ax, [joint2], markersize=12, color=:orange, label="Actuated Joint")
    scatter!(ax, [tip], markersize=10, color=:red, label="Tip")
    
    # Draw goal line
    goal_line = _goal_height_line(problem)
    lines!(ax, goal_line, linewidth=3, color=:green, linestyle=:dash, label="Goal Height")
    
    # Draw torque arrow if there's significant torque
    torque_data = _torque_arrow_coords(problem)
    if !isnothing(torque_data)
        arrows!(ax, [torque_data.pos], [Vec2f(torque_data.dx, torque_data.dy)],
            color=torque_data.color, arrowsize=12, linewidth=3)
    end
    
    # Set axis properties
    total_length = problem.link_length_1 + problem.link_length_2
    margin = 0.5f0
    xlims!(ax, -(total_length + margin), total_length + margin)
    ylims!(ax, -(total_length + margin), total_length + margin)
    
    ax.xlabel = "X Position"
    ax.ylabel = "Y Position"
    ax.title = "Acrobot Environment"
    
    # Add legend
    axislegend(ax, position=:rt)
    
    fig
end

function ClassicControlEnvironments.live_viz(problem::AcrobotProblem; size=(600, 600))
    theta1 = Observable(problem.theta1)
    theta2 = Observable(problem.theta2)
    torque = Observable(problem.torque)
    
    fig = Figure(size=size)
    ax = Axis(fig[1, 1], aspect=DataAspect())
    
    # Dynamic link positions
    joint1 = Observable(Point2f(0, 0))
    joint2 = @lift begin
        l1 = problem.link_length_1
        x = -l1 * cos($theta1)
        y = l1 * sin($theta1)
        Point2f(x, y)
    end
    tip = @lift begin
        l1 = problem.link_length_1
        l2 = problem.link_length_2
        x1 = -l1 * cos($theta1)
        y1 = l1 * sin($theta1)
        x2 = x1 - l2 * cos($theta1 + $theta2)
        y2 = y1 + l2 * sin($theta1 + $theta2)
        Point2f(x2, y2)
    end
    
    # Draw links (dynamic)
    link1_line = lines!(ax, @lift([$joint1, $joint2]), linewidth=8, color=:blue)
    link2_line = lines!(ax, @lift([$joint2, $tip]), linewidth=8, color=:blue)
    
    # Draw joints (dynamic)
    joint1_scatter = scatter!(ax, joint1, markersize=15, color=:black)
    joint2_scatter = scatter!(ax, joint2, markersize=12, color=:orange)
    tip_scatter = scatter!(ax, tip, markersize=10, color=:red)
    
    # Draw goal line (static)
    goal_line = _goal_height_line(problem)
    lines!(ax, goal_line, linewidth=3, color=:green, linestyle=:dash)
    
    # Draw torque arrow (dynamic)
    torque_arrow = arrows!(ax,
        @lift([$joint2]),
        @lift(begin
            t = $torque
            if abs(t) < 1e-3
                [Vec2f(0, 0)]
            else
                arrow_length = 0.3f0 * abs(t)
                angle_offset = t > 0 ? π/2 : -π/2
                [Vec2f(arrow_length * cos(angle_offset), arrow_length * sin(angle_offset))]
            end
        end),
        color=@lift($torque > 0 ? :green : :red),
        arrowsize=12,
        linewidth=3,
        visible=@lift(abs($torque) > 1e-3)
    )
    
    # Set axis properties
    total_length = problem.link_length_1 + problem.link_length_2
    margin = 0.5f0
    xlims!(ax, -(total_length + margin), total_length + margin)
    ylims!(ax, -(total_length + margin), total_length + margin)
    
    ax.xlabel = "X Position"
    ax.ylabel = "Y Position"
    ax.title = "Acrobot Environment"
    
    # Update function
    update_viz! = (problem) -> begin
        theta1[] = problem.theta1
        theta2[] = problem.theta2
        torque[] = problem.torque
    end
    
    return theta1, theta2, torque, fig, update_viz!
end

function ClassicControlEnvironments.interactive_viz(env::AcrobotEnv)
    theta1 = Observable(env.problem.theta1)
    theta2 = Observable(env.problem.theta2)
    torque = Observable(env.problem.torque)
    rew = Observable(reward(env))
    total_rew = Observable(0.0f0)
    auto_running = Observable(false)
    
    fig = Figure(size=(800, 700))
    ax = Axis(fig[1, 1], aspect=DataAspect())
    
    # Dynamic link positions
    joint1 = Observable(Point2f(0, 0))
    joint2 = @lift begin
        l1 = env.problem.link_length_1
        x = -l1 * cos($theta1)
        y = l1 * sin($theta1)
        Point2f(x, y)
    end
    tip = @lift begin
        l1 = env.problem.link_length_1
        l2 = env.problem.link_length_2
        x1 = -l1 * cos($theta1)
        y1 = l1 * sin($theta1)
        x2 = x1 - l2 * cos($theta1 + $theta2)
        y2 = y1 + l2 * sin($theta1 + $theta2)
        Point2f(x2, y2)
    end
    
    # Draw links
    link1_line = lines!(ax, @lift([$joint1, $joint2]), linewidth=8, color=:blue)
    link2_line = lines!(ax, @lift([$joint2, $tip]), linewidth=8, color=:blue)
    
    # Draw joints
    joint1_scatter = scatter!(ax, joint1, markersize=15, color=:black)
    joint2_scatter = scatter!(ax, joint2, markersize=12, color=:orange)
    tip_scatter = scatter!(ax, tip, markersize=10, color=:red)
    
    # Draw goal line
    goal_line = _goal_height_line(env.problem)
    lines!(ax, goal_line, linewidth=3, color=:green, linestyle=:dash)
    
    # Draw torque arrow
    torque_arrow = arrows!(ax,
        @lift([$joint2]),
        @lift(begin
            t = $torque
            if abs(t) < 1e-3
                [Vec2f(0, 0)]
            else
                arrow_length = 0.3f0 * abs(t)
                angle_offset = t > 0 ? π/2 : -π/2
                [Vec2f(arrow_length * cos(angle_offset), arrow_length * sin(angle_offset))]
            end
        end),
        color=@lift($torque > 0 ? :green : :red),
        arrowsize=12,
        linewidth=3,
        visible=@lift(abs($torque) > 1e-3)
    )
    
    # Set axis properties
    total_length = env.problem.link_length_1 + env.problem.link_length_2
    margin = 0.5f0
    xlims!(ax, -(total_length + margin), total_length + margin)
    ylims!(ax, -(total_length + margin), total_length + margin)
    
    ax.xlabel = "X Position"
    ax.ylabel = "Y Position"
    ax.title = "Acrobot Environment"
    
    # Reward display
    rew_ax = Axis(fig[1, 2], title="Cumulative Reward")
    rew_bar = barplot!(rew_ax, [1], @lift([$total_rew]), color=:blue)
    colsize!(fig.layout, 2, Relative(0.25))
    
    # Control buttons for actions
    button_grid = GridLayout(fig[2, 1:2])
    action_buttons = [
        Button(button_grid[1, 1], label="Torque -1", tellwidth=false),
        Button(button_grid[1, 2], label="Torque 0", tellwidth=false),
        Button(button_grid[1, 3], label="Torque +1", tellwidth=false)
    ]
    
    # Control buttons for simulation
    control_grid = GridLayout(fig[3, 1:2])
    start_button = Button(control_grid[1, 1], label="Start Auto", tellwidth=false)
    stop_button = Button(control_grid[1, 2], label="Stop Auto", tellwidth=false)
    step_button = Button(control_grid[1, 3], label="Single Step", tellwidth=false)
    reset_button = Button(control_grid[1, 4], label="Reset", tellwidth=false)
    
    current_task = Ref{Union{Task,Nothing}}(nothing)
    current_action = Observable(1)  # Default to no torque
    
    # Action button functionality
    for (i, button) in enumerate(action_buttons)
        on(button.clicks) do n
            if !auto_running[]
                current_action[] = i - 1  # Convert to 0-based action
                act!(env, current_action[])
                theta1[] = env.problem.theta1
                theta2[] = env.problem.theta2
                torque[] = env.problem.torque
                rew[] = reward(env)
                total_rew[] += rew[]
            end
        end
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
                        sleep(0.1)  # ~10 FPS
                        if auto_running[]
                            # Use current action
                            act!(env, current_action[])
                            theta1[] = env.problem.theta1
                            theta2[] = env.problem.theta2
                            torque[] = env.problem.torque
                            rew[] = reward(env)
                            total_rew[] += rew[]
                            
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
            act!(env, current_action[])
            theta1[] = env.problem.theta1
            theta2[] = env.problem.theta2
            torque[] = env.problem.torque
            rew[] = reward(env)
            total_rew[] += rew[]
        end
    end
    
    # Reset button
    on(reset_button.clicks) do n
        if !auto_running[]
            reset!(env)
            theta1[] = env.problem.theta1
            theta2[] = env.problem.theta2
            torque[] = env.problem.torque
            rew[] = reward(env)
            total_rew[] = 0.0f0
            current_action[] = 1  # Reset to no torque
            start_button.label = "Start Auto"
            start_button.buttoncolor = :lightgray
        end
    end
    
    display(fig)
    
    return theta1, theta2, torque, fig, action_buttons, start_button, stop_button, step_button, reset_button
end

function ClassicControlEnvironments.plot_trajectory(env::AcrobotEnv, observations::AbstractArray, actions::AbstractArray, rewards::AbstractArray)
    fig = Figure(size=(1000, 800))
    n = length(observations)
    
    # Extract observation components
    cos_theta1 = getindex.(observations, 1)
    sin_theta1 = getindex.(observations, 2)
    cos_theta2 = getindex.(observations, 3)
    sin_theta2 = getindex.(observations, 4)
    dtheta1 = getindex.(observations, 5)
    dtheta2 = getindex.(observations, 6)
    
    # Reconstruct angles
    theta1_vals = atan.(sin_theta1, cos_theta1)
    theta2_vals = atan.(sin_theta2, cos_theta2)
    
    # Convert actions to torque values
    if !isempty(actions) && actions[1] isa AbstractArray
        actions = [Int(a[1]) for a in actions]
    else
        actions = [Int(a) for a in actions]
    end
    torque_vals = [env.problem.avail_torque[a + 1] for a in actions]
    
    # Angle plots
    ax_theta1 = Axis(fig[1, 1], title="Link 1 Angle (θ₁)")
    scatterlines!(ax_theta1, rad2deg.(theta1_vals), label="θ₁ (degrees)")
    
    ax_theta2 = Axis(fig[1, 2], title="Link 2 Angle (θ₂)")
    scatterlines!(ax_theta2, rad2deg.(theta2_vals), label="θ₂ (degrees)")
    
    # Angular velocity plots
    ax_dtheta1 = Axis(fig[2, 1], title="Link 1 Angular Velocity")
    scatterlines!(ax_dtheta1, dtheta1, label="θ̇₁ (rad/s)")
    hlines!(ax_dtheta1, [-env.problem.max_vel_1, env.problem.max_vel_1], 
           color=:red, linestyle=:dash, label="Velocity Bounds")
    
    ax_dtheta2 = Axis(fig[2, 2], title="Link 2 Angular Velocity")
    scatterlines!(ax_dtheta2, dtheta2, label="θ̇₂ (rad/s)")
    hlines!(ax_dtheta2, [-env.problem.max_vel_2, env.problem.max_vel_2], 
           color=:red, linestyle=:dash, label="Velocity Bounds")
    
    # Action/torque plot
    ax_action = Axis(fig[3, 1], title="Applied Torque")
    scatterlines!(ax_action, torque_vals, label="Torque")
    hlines!(ax_action, [-1.0, 0.0, 1.0], color=:gray, linestyle=:dot, label="Available Torques")
    
    # Reward plot
    ax_rew = Axis(fig[3, 2], title="Rewards")
    scatterlines!(ax_rew, rewards, label="Reward")
    
    # Tip height over time
    ax_height = Axis(fig[4, 1:2], title="Tip Height Over Time")
    tip_heights = [-cos.(theta1_vals) .- cos.(theta2_vals .+ theta1_vals)]
    scatterlines!(ax_height, tip_heights, label="Tip Height")
    hlines!(ax_height, [1.0], color=:green, linestyle=:dash, label="Goal Height")
    ax_height.xlabel = "Time Step"
    ax_height.ylabel = "Height"
    
    # Add legends
    for ax in [ax_theta1, ax_theta2, ax_dtheta1, ax_dtheta2, ax_action, ax_rew, ax_height]
        axislegend(ax)
    end
    
    fig
end

function ClassicControlEnvironments.plot_trajectory_interactive(env::AcrobotEnv, observations::AbstractArray, actions::AbstractArray, rewards::AbstractArray)
    # Convert actions to torque values for visualization
    if !isempty(actions) && actions[1] isa AbstractArray
        actions = [Int(a[1]) for a in actions]
    else
        actions = [Int(a) for a in actions]
    end
    torque_vals = [env.problem.avail_torque[a + 1] for a in actions]
    
    num_steps = length(observations)
    if num_steps == 0
        error("Observations array cannot be empty.")
    end
    
    # Initial state for visualization
    initial_obs = observations[1]
    initial_theta1 = atan(initial_obs[2], initial_obs[1])
    initial_theta2 = atan(initial_obs[4], initial_obs[3])
    initial_torque = torque_vals[1]
    
    # Create problem for visualization
    problem_for_viz = AcrobotProblem(
        theta1=initial_theta1,
        theta2=initial_theta2,
        torque=initial_torque,
        dt=env.problem.dt,
        link_length_1=env.problem.link_length_1,
        link_length_2=env.problem.link_length_2,
        link_mass_1=env.problem.link_mass_1,
        link_mass_2=env.problem.link_mass_2,
        max_vel_1=env.problem.max_vel_1,
        max_vel_2=env.problem.max_vel_2
    )
    
    # Get live visualization
    _, _, _, fig, update_viz! = live_viz(problem_for_viz; size=(700, 600))
    
    # Add trajectory controls
    display(fig)
    sg = SliderGrid(fig[2, 1],
        (label="Step", range=1:num_steps, startvalue=1),
        (label="Playback Speed", range=0.01:0.01:0.2, startvalue=0.1)
    )
    trajectory_slider = sg.sliders[1]
    speed_slider = sg.sliders[2]
    
    # Control buttons
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
        current_theta1 = atan(current_obs[2], current_obs[1])
        current_theta2 = atan(current_obs[4], current_obs[3])
        current_torque = torque_vals[step_idx]
        
        updated_problem = AcrobotProblem(
            theta1=current_theta1,
            theta2=current_theta2,
            torque=current_torque,
            dt=env.problem.dt,
            link_length_1=env.problem.link_length_1,
            link_length_2=env.problem.link_length_2
        )
        update_viz!(updated_problem)
    end
    
    # Manual slider control
    on(trajectory_slider.value) do step_idx
        if !auto_playing[]
            update_step!(step_idx)
        end
    end
    
    # Start/Play button functionality
    on(start_button.clicks) do n
        if !auto_playing[]
            auto_playing[] = true
            start_button.label = "Playing..."
            start_button.buttoncolor = :lightgreen
            
            current_task[] = @async begin
                try
                    current_step = trajectory_slider.value[]
                    while auto_playing[] && current_step <= num_steps
                        sleep(speed_slider.value[])
                        if auto_playing[]
                            set_close_to!(trajectory_slider, current_step)
                            update_step!(current_step)
                            current_step += 1
                            
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
        end
    end
    
    # Single step button functionality
    on(step_button.clicks) do n
        if !auto_playing[]
            current_step = min(trajectory_slider.value[] + 1, num_steps)
            trajectory_slider.value[] = current_step
            notify(trajectory_slider)
            update_step!(current_step)
        end
    end
    
    # Reset button functionality
    on(reset_button.clicks) do n
        if !auto_playing[]
            trajectory_slider.value[] = 1
            notify(trajectory_slider)
            update_step!(1)
        end
    end
    
    return fig, trajectory_slider, start_button, stop_button, step_button, reset_button
end

function ClassicControlEnvironments.animate_trajectory_video(env::AcrobotEnv,
    observations::AbstractArray,
    actions::AbstractArray,
    output_filename::AbstractString;
    target_fps::Int=25
)
    # Convert actions to torque values
    if !isempty(actions) && actions[1] isa AbstractArray
        actions = [Int(a[1]) for a in actions]
    else
        actions = [Int(a) for a in actions]
    end
    torque_vals = [env.problem.avail_torque[a + 1] for a in actions]
    
    num_steps = length(observations)
    if num_steps == 0
        error("Observations array cannot be empty.")
    end
    
    # Initial state for visualization
    initial_obs = observations[1]
    initial_theta1 = atan(initial_obs[2], initial_obs[1])
    initial_theta2 = atan(initial_obs[4], initial_obs[3])
    
    problem_for_viz = AcrobotProblem(
        theta1=initial_theta1,
        theta2=initial_theta2,
        torque=torque_vals[1],
        dt=env.problem.dt,
        link_length_1=env.problem.link_length_1,
        link_length_2=env.problem.link_length_2
    )
    
    _, _, _, fig, update_viz! = live_viz(problem_for_viz)
    
    # Animation function
    function frame_update(step_idx)
        current_obs = observations[step_idx]
        current_theta1 = atan(current_obs[2], current_obs[1])
        current_theta2 = atan(current_obs[4], current_obs[3])
        current_torque = torque_vals[step_idx]
        
        updated_problem = AcrobotProblem(
            theta1=current_theta1,
            theta2=current_theta2,
            torque=current_torque,
            dt=env.problem.dt,
            link_length_1=env.problem.link_length_1,
            link_length_2=env.problem.link_length_2
        )
        update_viz!(updated_problem)
    end
    
    # Use dt from problem for frame rate calculation
    dt = env.problem.dt
    steps_per_frame = max(1, round(Int, 1 / (target_fps * dt)))
    frame_indices = 1:steps_per_frame:num_steps
    
    Makie.record(fig, output_filename, frame_indices; framerate=target_fps) do step_idx
        frame_update(step_idx)
    end
    
    return output_filename
end

function ClassicControlEnvironments.plot_trajectory_phase_space(env::AcrobotEnv, observations::AbstractArray, actions::AbstractArray; size=(800, 600))
    # Extract observation components and reconstruct angles
    cos_theta1 = getindex.(observations, 1)
    sin_theta1 = getindex.(observations, 2)
    cos_theta2 = getindex.(observations, 3)
    sin_theta2 = getindex.(observations, 4)
    dtheta1 = getindex.(observations, 5)
    dtheta2 = getindex.(observations, 6)
    
    theta1_vals = atan.(sin_theta1, cos_theta1)
    theta2_vals = atan.(sin_theta2, cos_theta2)
    
    fig = Figure(size=size)
    
    # Phase space for link 1
    ax1 = Axis(fig[1, 1], title="Link 1 Phase Space (θ₁ vs θ̇₁)")
    scatterlines!(ax1, rad2deg.(theta1_vals), dtheta1, label="Trajectory")
    scatter!(ax1, [rad2deg(theta1_vals[1])], [dtheta1[1]], color=:green, markersize=10, label="Start")
    scatter!(ax1, [rad2deg(theta1_vals[end])], [dtheta1[end]], color=:red, markersize=10, label="End")
    hlines!(ax1, [-env.problem.max_vel_1, env.problem.max_vel_1], color=:red, linestyle=:dash, label="Velocity Bounds")
    ax1.xlabel = "θ₁ (degrees)"
    ax1.ylabel = "θ̇₁ (rad/s)"
    axislegend(ax1)
    
    # Phase space for link 2
    ax2 = Axis(fig[1, 2], title="Link 2 Phase Space (θ₂ vs θ̇₂)")
    scatterlines!(ax2, rad2deg.(theta2_vals), dtheta2, label="Trajectory")
    scatter!(ax2, [rad2deg(theta2_vals[1])], [dtheta2[1]], color=:green, markersize=10, label="Start")
    scatter!(ax2, [rad2deg(theta2_vals[end])], [dtheta2[end]], color=:red, markersize=10, label="End")
    hlines!(ax2, [-env.problem.max_vel_2, env.problem.max_vel_2], color=:red, linestyle=:dash, label="Velocity Bounds")
    ax2.xlabel = "θ₂ (degrees)"
    ax2.ylabel = "θ̇₂ (rad/s)"
    axislegend(ax2)
    
    # Combined configuration space
    ax3 = Axis(fig[2, 1:2], title="Configuration Space (θ₁ vs θ₂)")
    scatterlines!(ax3, rad2deg.(theta1_vals), rad2deg.(theta2_vals), label="Trajectory")
    scatter!(ax3, [rad2deg(theta1_vals[1])], [rad2deg(theta2_vals[1])], color=:green, markersize=10, label="Start")
    scatter!(ax3, [rad2deg(theta1_vals[end])], [rad2deg(theta2_vals[end])], color=:red, markersize=10, label="End")
    ax3.xlabel = "θ₁ (degrees)"
    ax3.ylabel = "θ₂ (degrees)"
    axislegend(ax3)
    
    return fig
end 