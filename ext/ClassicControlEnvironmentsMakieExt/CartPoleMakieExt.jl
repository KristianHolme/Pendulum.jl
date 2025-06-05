function _cart_position(problem::CartPoleProblem)
    # Cart position at ground level
    return Point2f(problem.x, 0.0f0)
end

function _pole_tip_position(problem::CartPoleProblem)
    # Pole tip position - pole extends upward from cart center
    # theta = 0 is upright, positive theta is clockwise
    cart_x = problem.x
    pole_length = 2 * problem.length  # Full pole length (length is half-pole)
    pole_tip_x = cart_x + pole_length * sin(problem.theta)
    pole_tip_y = pole_length * cos(problem.theta)
    return Point2f(pole_tip_x, pole_tip_y)
end

function _track_bounds(problem::CartPoleProblem)
    # Track extends beyond the cart bounds for visualization
    margin = 0.5f0
    left = -problem.x_threshold - margin
    right = problem.x_threshold + margin
    return (left, right)
end

function _force_arrow_coords(problem::CartPoleProblem)
    cart_pos = _cart_position(problem)
    force = problem.force
    
    # Arrow length scales with force magnitude
    arrow_length = 0.3 * abs(force) / problem.force_mag
    dx = arrow_length * sign(force)  # Force direction: positive = right, negative = left
    dy = 0.0
    
    # Color based on force direction
    color = force > 0 ? :green : :red  # Green for right, red for left
    
    (; cart_pos, dx, dy, color, force)
end

function ClassicControlEnvironments.plot(problem::CartPoleProblem)
    fig = Figure(size=(600, 500))
    ax = Axis(fig[1, 1], aspect=DataAspect())
    
    # Draw track
    track_left, track_right = _track_bounds(problem)
    track_y = -0.1f0
    lines!(ax, [track_left, track_right], [track_y, track_y], linewidth=4, color=:black)
    
    # Draw cart as rectangle
    cart_pos = _cart_position(problem)
    cart_width = 0.2f0
    cart_height = 0.1f0
    cart_rect = Rect2f(cart_pos[1] - cart_width/2, cart_pos[2] - cart_height/2, cart_width, cart_height)
    poly!(ax, cart_rect, color=:blue, strokecolor=:black, strokewidth=2)
    
    # Draw pole
    pole_tip = _pole_tip_position(problem)
    lines!(ax, [cart_pos, pole_tip], linewidth=4, color=:brown)
    
    # Draw pole mass (circle at tip)
    scatter!(ax, pole_tip, color=:red, markersize=15)
    
    # Draw cart center (pivot point)
    scatter!(ax, cart_pos, color=:black, markersize=8)
    
    # Draw force arrow if there's significant force
    if abs(problem.force) > 1e-3
        arrow_data = _force_arrow_coords(problem)
        arrows!(ax, [arrow_data.cart_pos], [Vec2f(arrow_data.dx, arrow_data.dy)],
            color=arrow_data.color, arrowsize=15, linewidth=3)
    end
    
    # Draw bounds indicators
    vlines!(ax, [-problem.x_threshold, problem.x_threshold], color=:red, linestyle=:dash, alpha=0.7, linewidth=2)
    
    # Labels and formatting
    ax.xlabel = "Position"
    ax.ylabel = "Height" 
    ax.title = "CartPole Environment"
    
    xlims!(ax, track_left, track_right)
    ylims!(ax, -0.3, 1.2)
    
    fig
end

function ClassicControlEnvironments.live_viz(problem::CartPoleProblem; size=(600, 500))
    x = Observable(problem.x)
    theta = Observable(problem.theta)
    force = Observable(problem.force)
    
    fig = Figure(size=size)
    ax = Axis(fig[1, 1], aspect=DataAspect())
    
    # Track (static)
    track_left, track_right = _track_bounds(problem)
    track_y = -0.1f0
    lines!(ax, [track_left, track_right], [track_y, track_y], linewidth=4, color=:black)
    
    # Cart (dynamic)
    cart_width = 0.2f0
    cart_height = 0.1f0
    cart_rect = poly!(ax, 
        @lift(Rect2f($x - cart_width/2, -cart_height/2, cart_width, cart_height)),
        color=:blue, strokecolor=:black, strokewidth=2)
    
    # Pole (dynamic)
    pole_length = 2 * problem.length
    pole_line = lines!(ax,
        @lift([Point2f($x, 0.0f0), Point2f($x + pole_length * sin($theta), pole_length * cos($theta))]),
        linewidth=4, color=:brown)
    
    # Pole mass (dynamic)
    pole_mass = scatter!(ax,
        @lift(Point2f($x + pole_length * sin($theta), pole_length * cos($theta))),
        color=:red, markersize=15)
    
    # Cart center (dynamic)
    cart_center = scatter!(ax, @lift(Point2f($x, 0.0f0)), color=:black, markersize=8)
    
    # Force arrow (dynamic)
    force_arrow = arrows!(ax,
        @lift([Point2f($x, 0.0f0)]),
        @lift([Vec2f(0.3 * abs($force) / problem.force_mag * sign($force), 0.0)]),
        color=@lift($force > 0 ? :green : :red),
        arrowsize=15,
        linewidth=3,
        visible=@lift(abs($force) > 1e-3)
    )
    
    # Bounds indicators (static)
    vlines!(ax, [-problem.x_threshold, problem.x_threshold], color=:red, linestyle=:dash, alpha=0.7, linewidth=2)
    
    # Labels and formatting
    ax.xlabel = "Position"
    ax.ylabel = "Height"
    ax.title = "CartPole Environment"
    
    xlims!(ax, track_left, track_right)
    ylims!(ax, -0.3, 1.2)
    
    # Update function
    update_viz! = (problem) -> begin
        x[] = problem.x
        theta[] = problem.theta  
        force[] = problem.force
    end
    
    return x, theta, force, fig, update_viz!
end

function ClassicControlEnvironments.interactive_viz(env::CartPoleEnv)
    x = Observable(env.problem.x)
    theta = Observable(env.problem.theta)
    force = Observable(env.problem.force)
    rew = Observable(reward(env))
    total_rew = Observable(reward(env))
    auto_running = Observable(false)
    
    fig = Figure(size=(800, 700))
    ax = Axis(fig[1, 1], aspect=DataAspect())
    
    # Track
    track_left, track_right = _track_bounds(env.problem)
    track_y = -0.1f0
    lines!(ax, [track_left, track_right], [track_y, track_y], linewidth=4, color=:black)
    
    # Cart
    cart_width = 0.2f0
    cart_height = 0.1f0
    cart_rect = poly!(ax,
        @lift(Rect2f($x - cart_width/2, -cart_height/2, cart_width, cart_height)),
        color=:blue, strokecolor=:black, strokewidth=2)
    
    # Pole
    pole_length = 2 * env.problem.length
    pole_line = lines!(ax,
        @lift([Point2f($x, 0.0f0), Point2f($x + pole_length * sin($theta), pole_length * cos($theta))]),
        linewidth=4, color=:brown)
    
    # Pole mass
    pole_mass = scatter!(ax,
        @lift(Point2f($x + pole_length * sin($theta), pole_length * cos($theta))),
        color=:red, markersize=15)
    
    # Cart center
    cart_center = scatter!(ax, @lift(Point2f($x, 0.0f0)), color=:black, markersize=8)
    
    # Force arrow
    force_arrow = arrows!(ax,
        @lift([Point2f($x, 0.0f0)]),
        @lift([Vec2f(0.3 * abs($force) / env.problem.force_mag * sign($force), 0.0)]),
        color=@lift($force > 0 ? :green : :red),
        arrowsize=15,
        linewidth=3,
        visible=@lift(abs($force) > 1e-3)
    )
    
    # Bounds indicators
    vlines!(ax, [-env.problem.x_threshold, env.problem.x_threshold], color=:red, linestyle=:dash, alpha=0.7, linewidth=2)
    
    ax.xlabel = "Position"
    ax.ylabel = "Height"
    ax.title = "CartPole Environment"
    xlims!(ax, track_left, track_right)
    ylims!(ax, -0.3, 1.2)
    
    # Reward displays
    rew_ax = Axis(fig[1, 2], title="Step Reward")
    rew_bar = barplot!(rew_ax, 1, rew, color=@lift($rew > 0 ? :green : :red))
    ylims!(rew_ax, -1.5, 1.5)
    
    total_rew_ax = Axis(fig[1, 3], title="Total Reward")
    total_rew_bar = barplot!(total_rew_ax, 1, total_rew, color=:blue)
    
    colsize!(fig.layout, 2, Relative(0.15))
    colsize!(fig.layout, 3, Relative(0.15))
    
    # Control buttons (discrete actions)
    button_grid = GridLayout(fig[2, 1])
    left_button = Button(button_grid[1, 1], label="Push Left (0)", tellwidth=false)
    right_button = Button(button_grid[1, 2], label="Push Right (1)", tellwidth=false)
    auto_left_button = Button(button_grid[1, 3], label="Auto Left", tellwidth=false)
    auto_right_button = Button(button_grid[1, 4], label="Auto Right", tellwidth=false)
    stop_button = Button(button_grid[1, 5], label="Stop", tellwidth=false)
    reset_button = Button(button_grid[1, 6], label="Reset", tellwidth=false)
    
    current_task = Ref{Union{Task,Nothing}}(nothing)
    
    # Single action buttons
    on(left_button.clicks) do n
        if !auto_running[]
            act!(env, 0)  # Push left
            x[] = env.problem.x
            theta[] = env.problem.theta
            force[] = env.problem.force
            rew[] = reward(env)
            total_rew[] += rew[]
        end
    end
    
    on(right_button.clicks) do n
        if !auto_running[]
            act!(env, 1)  # Push right
            x[] = env.problem.x
            theta[] = env.problem.theta
            force[] = env.problem.force
            rew[] = reward(env)
            total_rew[] += rew[]
        end
    end
    
    # Auto action functions
    function start_auto_action(action)
        if !auto_running[]
            auto_running[] = true
            
            current_task[] = @async begin
                try
                    while auto_running[]
                        sleep(env.problem.tau)  # Use environment timestep
                        if auto_running[]
                            act!(env, action)
                            x[] = env.problem.x
                            theta[] = env.problem.theta
                            force[] = env.problem.force
                            rew[] = reward(env)
                            total_rew[] += rew[]
                            
                            # Check if episode ended
                            if terminated(env) || truncated(env)
                                auto_running[] = false
                                break
                            end
                        end
                    end
                catch e
                    @warn "Auto-action task interrupted: $e"
                finally
                    auto_running[] = false
                    auto_left_button.buttoncolor = :lightgray
                    auto_right_button.buttoncolor = :lightgray
                end
            end
        end
    end
    
    on(auto_left_button.clicks) do n
        auto_left_button.buttoncolor = :lightcoral
        auto_right_button.buttoncolor = :lightgray
        start_auto_action(0)
    end
    
    on(auto_right_button.clicks) do n
        auto_right_button.buttoncolor = :lightgreen
        auto_left_button.buttoncolor = :lightgray
        start_auto_action(1)
    end
    
    # Stop button
    on(stop_button.clicks) do n
        if auto_running[]
            auto_running[] = false
            auto_left_button.buttoncolor = :lightgray
            auto_right_button.buttoncolor = :lightgray
        end
    end
    
    # Reset button
    on(reset_button.clicks) do n
        if !auto_running[]
            reset!(env)
            x[] = env.problem.x
            theta[] = env.problem.theta
            force[] = 0.0f0
            rew[] = reward(env)
            total_rew[] = rew[]
            auto_left_button.buttoncolor = :lightgray
            auto_right_button.buttoncolor = :lightgray
        end
    end
    
    display(fig)
    
    return x, theta, force, fig, button_grid, left_button, right_button, auto_left_button, auto_right_button, stop_button, reset_button
end

function ClassicControlEnvironments.plot_trajectory(env::CartPoleEnv, observations::AbstractArray, actions::AbstractArray, rewards::AbstractArray)
    fig = Figure(size=(1000, 800))
    n = length(observations)
    
    positions = getindex.(observations, 1)
    velocities = getindex.(observations, 2)
    angles = getindex.(observations, 3)
    angular_velocities = getindex.(observations, 4)
    
    # Convert actions to integers if needed
    if !isempty(actions) && actions[1] isa AbstractArray
        actions = [Int(a[1]) for a in actions]
    else
        actions = [Int(a) for a in actions]
    end
    
    # Position over time
    ax_pos = Axis(fig[1, 1], title="Cart Position over Time")
    pos_line = scatterlines!(ax_pos, positions, label="Position")
    hlines!(ax_pos, [-env.problem.x_threshold, env.problem.x_threshold], color=:red, linestyle=:dash, label="Position Bounds")
    axislegend(ax_pos)
    
    # Velocity over time
    ax_vel = Axis(fig[1, 2], title="Cart Velocity over Time")
    vel_line = scatterlines!(ax_vel, velocities, label="Velocity")
    
    # Pole angle over time
    ax_angle = Axis(fig[2, 1], title="Pole Angle over Time")
    angle_line = scatterlines!(ax_angle, rad2deg.(angles), label="Angle (degrees)")
    hlines!(ax_angle, rad2deg.([-env.problem.theta_threshold_radians, env.problem.theta_threshold_radians]), 
           color=:red, linestyle=:dash, label="Angle Bounds")
    axislegend(ax_angle)
    
    # Pole angular velocity over time
    ax_angvel = Axis(fig[2, 2], title="Pole Angular Velocity over Time") 
    angvel_line = scatterlines!(ax_angvel, angular_velocities, label="Angular Velocity")
    
    # Actions over time
    ax_action = Axis(fig[3, 1], title="Actions over Time")
    action_line = scatterlines!(ax_action, actions, label="Action")
    hlines!(ax_action, [0, 1], color=:gray, linestyle=:dot, label="Action Range")
    ax_action.yticks = [0, 1]
    ax_action.yticklabels = ["Push Left", "Push Right"]
    axislegend(ax_action)
    
    # Rewards over time
    ax_rew = Axis(fig[3, 2], title="Rewards over Time")
    rew_line = scatterlines!(ax_rew, rewards, label="Reward")
    axislegend(ax_rew)
    
    fig
end

function ClassicControlEnvironments.plot_trajectory_interactive(env::CartPoleEnv, observations::AbstractArray, actions::AbstractArray, rewards::AbstractArray)
    # Process actions: ensure they are integers
    local processed_actions::Vector{Int}
    if !isempty(actions) && actions[1] isa AbstractArray
        processed_actions = [Int(a[1]) for a in actions]
    else
        processed_actions = [Int(a) for a in actions]
    end
    
    num_steps = length(observations)
    if num_steps == 0
        error("Observations array cannot be empty.")
    end
    if num_steps != length(processed_actions)
        error("Observations and processed actions must have the same length.")
    end
    
    # Initial state for visualization
    initial_obs = observations[1]
    initial_x = initial_obs[1]
    initial_theta = initial_obs[3]
    
    # Create problem for visualization
    problem_for_viz = CartPoleProblem(
        x=Float32(initial_x),
        theta=Float32(initial_theta),
        force=processed_actions[1] == 0 ? -env.problem.force_mag : env.problem.force_mag,
        gravity=env.problem.gravity,
        masscart=env.problem.masscart,
        masspole=env.problem.masspole,
        length=env.problem.length,
        force_mag=env.problem.force_mag,
        tau=env.problem.tau,
        theta_threshold_radians=env.problem.theta_threshold_radians,
        x_threshold=env.problem.x_threshold
    )
    
    # Get live visualization
    _, _, _, fig, update_viz! = live_viz(problem_for_viz; size=(700, 600))
    
    # Add trajectory controls
    display(fig)
    sg = SliderGrid(fig[2, 1],
        (label="Step", range=1:num_steps, startvalue=1),
        (label="Playback Speed", range=0.01:0.01:0.1, startvalue=0.05)
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
        current_x = current_obs[1]
        current_theta = current_obs[3]
        current_action = processed_actions[step_idx]
        current_force = current_action == 0 ? -env.problem.force_mag : env.problem.force_mag
        
        updated_problem = CartPoleProblem(
            x=Float32(current_x),
            theta=Float32(current_theta),
            force=Float32(current_force),
            gravity=env.problem.gravity,
            masscart=env.problem.masscart,
            masspole=env.problem.masspole,
            length=env.problem.length,
            force_mag=env.problem.force_mag,
            tau=env.problem.tau,
            theta_threshold_radians=env.problem.theta_threshold_radians,
            x_threshold=env.problem.x_threshold
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
    
    # Stop/Pause button
    on(stop_button.clicks) do n
        if auto_playing[]
            auto_playing[] = false
            start_button.label = "Play"
            start_button.buttoncolor = :lightgray
        end
    end
    
    # Single step button
    on(step_button.clicks) do n
        if !auto_playing[]
            current_step = min(trajectory_slider.value[] + 1, num_steps)
            trajectory_slider.value[] = current_step
            notify(trajectory_slider)
            update_step!(current_step)
        end
    end
    
    # Reset button
    on(reset_button.clicks) do n
        if !auto_playing[]
            trajectory_slider.value[] = 1
            notify(trajectory_slider)
            update_step!(1)
        end
    end
    
    return fig, trajectory_slider, start_button, stop_button, step_button, reset_button
end

function ClassicControlEnvironments.animate_trajectory_video(env::CartPoleEnv,
    observations::AbstractArray,
    actions::AbstractArray,
    output_filename::AbstractString;
    target_fps::Int=25
)
    # Process actions
    if actions[1] isa AbstractArray
        actions = Int.(first.(actions))
    else
        actions = Int.(actions)
    end
    
    num_steps = length(observations)
    if num_steps == 0
        error("Observations array cannot be empty.")
    end
    if num_steps != length(actions)
        error("Observations and actions must have the same length.")
    end
    
    # Initial state
    initial_obs = observations[1]
    initial_x = initial_obs[1]
    initial_theta = initial_obs[3]
    initial_force = actions[1] == 0 ? -env.problem.force_mag : env.problem.force_mag
    
    problem_for_viz = env.problem
    problem_for_viz.x = initial_x
    problem_for_viz.theta = initial_theta
    problem_for_viz.force = initial_force
    
    _, _, _, fig, update_viz! = live_viz(problem_for_viz)
    
    # Animation function
    function frame_update(step_idx)
        current_obs = observations[step_idx]
        current_x = current_obs[1]
        current_theta = current_obs[3]
        current_action = actions[step_idx]
        current_force = current_action == 0 ? -env.problem.force_mag : env.problem.force_mag
        
        updated_problem = CartPoleProblem(
            x=Float32(current_x),
            theta=Float32(current_theta),
            force=Float32(current_force),
            gravity=env.problem.gravity,
            masscart=env.problem.masscart,
            masspole=env.problem.masspole,
            length=env.problem.length,
            force_mag=env.problem.force_mag,
            tau=env.problem.tau,
            theta_threshold_radians=env.problem.theta_threshold_radians,
            x_threshold=env.problem.x_threshold
        )
        update_viz!(updated_problem)
    end
    
    # Use environment timestep for frame rate calculation
    dt = env.problem.tau
    steps_per_frame = max(1, round(Int, 1 / (target_fps * dt)))
    frame_indices = 1:steps_per_frame:num_steps
    
    Makie.record(fig, output_filename, frame_indices; framerate=target_fps) do step_idx
        frame_update(step_idx)
    end
    
    return output_filename
end

function ClassicControlEnvironments.plot_trajectory_phase_space(env::CartPoleEnv, observations::AbstractArray, actions::AbstractArray; size=(800, 600))
    positions = getindex.(observations, 1)
    velocities = getindex.(observations, 2)
    angles = rad2deg.(getindex.(observations, 3))
    angular_velocities = getindex.(observations, 4)
    
    fig = Figure(size=size)
    
    # Position vs Velocity phase space
    ax1 = Axis(fig[1, 1], title="Cart Phase Space (Position vs Velocity)")
    traj_line1 = scatterlines!(ax1, positions, velocities, label="Trajectory")
    scatter!(ax1, [positions[1]], [velocities[1]], color=:green, markersize=10, label="Start")
    scatter!(ax1, [positions[end]], [velocities[end]], color=:red, markersize=10, label="End")
    vlines!(ax1, [-env.problem.x_threshold, env.problem.x_threshold], color=:red, linestyle=:dash, label="Position Bounds")
    ax1.xlabel = "Position"
    ax1.ylabel = "Velocity"
    axislegend(ax1)
    
    # Angle vs Angular Velocity phase space  
    ax2 = Axis(fig[1, 2], title="Pole Phase Space (Angle vs Angular Velocity)")
    traj_line2 = scatterlines!(ax2, angles, angular_velocities, label="Trajectory")
    scatter!(ax2, [angles[1]], [angular_velocities[1]], color=:green, markersize=10, label="Start")
    scatter!(ax2, [angles[end]], [angular_velocities[end]], color=:red, markersize=10, label="End")
    angle_bounds_deg = rad2deg(env.problem.theta_threshold_radians)
    vlines!(ax2, [-angle_bounds_deg, angle_bounds_deg], color=:red, linestyle=:dash, label="Angle Bounds")
    ax2.xlabel = "Angle (degrees)"
    ax2.ylabel = "Angular Velocity"
    axislegend(ax2)
    
    # Combined 4D visualization projected onto 2D
    ax3 = Axis(fig[2, 1:2], title="Combined State Trajectory")
    # Plot position trajectory with color coding for pole angle
    scatter!(ax3, positions, velocities, color=angles, colormap=:RdYlBu, markersize=6, label="Trajectory (colored by pole angle)")
    scatter!(ax3, [positions[1]], [velocities[1]], color=:green, markersize=15, marker=:star4, label="Start")
    scatter!(ax3, [positions[end]], [velocities[end]], color=:red, markersize=15, marker=:star4, label="End")
    vlines!(ax3, [-env.problem.x_threshold, env.problem.x_threshold], color=:red, linestyle=:dash, alpha=0.7)
    ax3.xlabel = "Cart Position"
    ax3.ylabel = "Cart Velocity"
    
    # Add colorbar for pole angle
    Colorbar(fig[2, 3], colormap=:RdYlBu, limits=extrema(angles), label="Pole Angle (degrees)")
    
    fig[3, 1:2] = Legend(fig, ax3, orientation=:horizontal)
    
    return fig
end 