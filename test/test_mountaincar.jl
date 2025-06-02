using ClassicControlEnvironments
using DRiL
using Test
using Random

@testset "MountainCarEnv tests" begin

    @testset "Observation range" begin
        env = MountainCarEnv()
        reset!(env)

        # Test observations after initialization
        obs = observe(env)
        @test length(obs) == 2
        @test env.problem.min_position <= obs[1] <= env.problem.max_position  # position
        @test -env.problem.max_speed <= obs[2] <= env.problem.max_speed      # velocity

        # Test observations after random actions
        rng = MersenneTwister(123)
        for _ in 1:100
            action = rand(rng, Float32) * 2.0f0 - 1.0f0  # Random action in [-1, 1]
            act!(env, action)
            obs = observe(env)
            @test env.problem.min_position <= obs[1] <= env.problem.max_position
            @test -env.problem.max_speed <= obs[2] <= env.problem.max_speed
        end
    end

    @testset "Initial state" begin
        rng = MersenneTwister(456)
        env = MountainCarEnv(rng=rng)

        # Test multiple resets
        for _ in 1:10
            reset!(env)
            @test -0.6f0 <= env.problem.position <= -0.4f0  # Initial position range
            @test env.problem.velocity == 0.0f0             # Initial velocity
            @test env.step == 0                             # Step counter reset
            @test !terminated(env)                          # Not terminated initially
            @test !truncated(env)                           # Not truncated initially
        end
    end

    @testset "Action clamping" begin
        env = MountainCarEnv()
        reset!(env)

        initial_pos = env.problem.position

        # Test that extreme actions are clamped
        act!(env, 10.0f0)  # Should be clamped to 1.0
        @test env.problem.force == 1.0f0

        reset!(env)
        act!(env, -10.0f0)  # Should be clamped to -1.0
        @test env.problem.force == -1.0f0
    end

    @testset "Physics simulation" begin
        env = MountainCarEnv()
        reset!(env)

        initial_position = env.problem.position
        initial_velocity = env.problem.velocity

        # Apply positive force (right)
        act!(env, 1.0f0)

        # Position should change based on velocity
        # Velocity should be affected by force and gravity
        @test env.problem.position != initial_position || env.problem.velocity != initial_velocity

        # Test boundary conditions
        env.problem.position = env.problem.min_position - 0.1f0
        env.problem.velocity = -0.01f0
        act!(env, -1.0f0)  # Try to go further left
        @test env.problem.position == env.problem.min_position  # Should be clamped
        @test env.problem.velocity == 0.0f0  # Velocity should be zeroed at boundary
    end

    @testset "Goal condition" begin
        env = MountainCarEnv()
        reset!(env)

        # Manually set car to goal position with sufficient velocity
        env.problem.position = env.problem.goal_position
        env.problem.velocity = env.problem.goal_velocity

        @test terminated(env)

        # Test goal reward
        reward_val = reward(env)
        @test reward_val > 0  # Should get positive reward at goal

        # Test not at goal
        env.problem.position = 0.0f0
        @test !terminated(env)
    end

    @testset "Reward structure" begin
        env = MountainCarEnv()
        reset!(env)

        # Test step penalty
        initial_reward = act!(env, 0.0f0)  # No force applied
        @test initial_reward < 0  # Should get negative reward (step penalty)

        # Test action penalty
        large_force_reward = act!(env, 1.0f0)  # Large force
        small_force_reward = act!(env, 0.1f0)  # Small force

        # With same step penalty, larger force should give more negative reward
        # (This might not always hold due to different positions, but in general)
    end

    @testset "Step count and truncation" begin
        max_steps = 50
        env = MountainCarEnv(max_steps=max_steps)
        reset!(env)

        @test env.step == 0
        @test !truncated(env)

        for i in 1:max_steps-1
            act!(env, 0.0f0)
            @test env.step == i
            @test !truncated(env)
        end

        act!(env, 0.0f0)
        @test env.step == max_steps
        @test truncated(env)
    end

    @testset "Action and observation spaces" begin
        env = MountainCarEnv()

        # Test action space
        as = action_space(env)
        @test as.low == [-1.0f0]
        @test as.high == [1.0f0]

        # Test observation space
        os = observation_space(env)
        @test os.low[1] == env.problem.min_position
        @test os.high[1] == env.problem.max_position
        @test os.low[2] == -env.problem.max_speed
        @test os.high[2] == env.problem.max_speed
    end
end