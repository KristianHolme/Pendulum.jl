using ClassicControlEnvironments
using DRiL
using Test
using Random

@testset "Mountain Car Environments" begin

    # Test both discrete and continuous versions
    discrete_env = MountainCarEnv()
    continuous_env = MountainCarContinuousEnv()
    
    @testset "Common functionality for $(typeof(env))" for env in [discrete_env, continuous_env]
        
        @testset "Observation range" begin
            reset!(env)

            # Test observations after initialization
            obs = observe(env)
            @test length(obs) == 2
            @test env.problem.min_position <= obs[1] <= env.problem.max_position  # position
            @test -env.problem.max_speed <= obs[2] <= env.problem.max_speed      # velocity

            # Test observations after actions (different for discrete vs continuous)
            rng = MersenneTwister(123)
            for _ in 1:20
                if env isa MountainCarEnv  # Discrete
                    action = rand(rng, 0:2)  # Random discrete action
                else  # Continuous
                    action = rand(rng, Float32) * 2.0f0 - 1.0f0  # Random action in [-1, 1]
                end
                act!(env, action)
                obs = observe(env)
                @test env.problem.min_position <= obs[1] <= env.problem.max_position
                @test -env.problem.max_speed <= obs[2] <= env.problem.max_speed
            end
        end

        @testset "Initial state" begin
            rng = MersenneTwister(456)
            if env isa MountainCarEnv
                test_env = MountainCarEnv(rng=rng)
            else
                test_env = MountainCarContinuousEnv(rng=rng)
            end

            # Test multiple resets
            for _ in 1:10
                reset!(test_env)
                @test -0.6f0 <= test_env.problem.position <= -0.4f0  # Initial position range
                @test test_env.problem.velocity == 0.0f0             # Initial velocity
                @test test_env.step == 0                             # Step counter reset
                @test !terminated(test_env)                          # Not terminated initially
                @test !truncated(test_env)                           # Not truncated initially
            end
        end

        @testset "Physics simulation" begin
            reset!(env)

            initial_position = env.problem.position
            initial_velocity = env.problem.velocity

            # Apply action (different for discrete vs continuous)
            if env isa MountainCarEnv  # Discrete
                act!(env, 2)  # Push right
                @test env.problem.force == 1.0f0  # Should map to +1 force
            else  # Continuous
                act!(env, 1.0f0)  # Push right
                @test env.problem.force == 1.0f0
            end

            # Position should change based on velocity
            # Velocity should be affected by force and gravity
            @test env.problem.position != initial_position || env.problem.velocity != initial_velocity

            # Test boundary conditions
            env.problem.position = env.problem.min_position - 0.1f0
            env.problem.velocity = -0.01f0
            if env isa MountainCarEnv  # Discrete
                act!(env, 0)  # Push left
            else  # Continuous
                act!(env, -1.0f0)  # Push left
            end
            @test env.problem.position == env.problem.min_position  # Should be clamped
            @test env.problem.velocity == 0.0f0  # Velocity should be zeroed at boundary
        end

        @testset "Goal condition" begin
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
            reset!(env)

            # Test step penalty with no force
            if env isa MountainCarEnv  # Discrete
                initial_reward = act!(env, 1)  # No force (action 1)
            else  # Continuous
                initial_reward = act!(env, 0.0f0)  # No force applied
            end
            @test initial_reward < 0  # Should get negative reward or zero
        end

        @testset "Step count and truncation" begin
            max_steps = 50
            if env isa MountainCarEnv
                test_env = MountainCarEnv(max_steps=max_steps)
            else
                test_env = MountainCarContinuousEnv(max_steps=max_steps)
            end
            reset!(test_env)

            @test test_env.step == 0
            @test !truncated(test_env)

            for i in 1:max_steps-1
                if test_env isa MountainCarEnv  # Discrete
                    act!(test_env, 1)  # No force
                else  # Continuous
                    act!(test_env, 0.0f0)  # No force
                end
                @test test_env.step == i
                @test !truncated(test_env)
            end

            if test_env isa MountainCarEnv  # Discrete
                act!(test_env, 1)  # No force
            else  # Continuous
                act!(test_env, 0.0f0)  # No force
            end
            @test test_env.step == max_steps
            @test truncated(test_env)
        end
    end

    @testset "Discrete MountainCarEnv specific tests" begin
        env = MountainCarEnv()
        reset!(env)

        @testset "Action space" begin
            as = action_space(env)
            @test as isa Discrete
            @test as.n == 3
            @test as.start == 0
            @test 0 ∈ as
            @test 1 ∈ as
            @test 2 ∈ as
            @test !(3 ∈ as)
        end

        @testset "Action mapping" begin
            reset!(env)
            
            # Test left action (0 -> -1 force)
            act!(env, 0)
            @test env.problem.force == -1.0f0
            
            reset!(env)
            # Test no action (1 -> 0 force)
            act!(env, 1)
            @test env.problem.force == 0.0f0
            
            reset!(env)
            # Test right action (2 -> +1 force) 
            act!(env, 2)
            @test env.problem.force == 1.0f0
        end

        @testset "Array action input" begin
            reset!(env)
            act!(env, [2])  # Should work with array input
            @test env.problem.force == 1.0f0
        end
    end

    @testset "Continuous MountainCarContinuousEnv specific tests" begin
        env = MountainCarContinuousEnv()
        reset!(env)

        @testset "Action space" begin
            as = action_space(env)
            @test as isa Box{Float32}
            @test as.low == [-1.0f0]
            @test as.high == [1.0f0]
        end

        @testset "Action clamping" begin
            reset!(env)

            # Test that extreme actions are clamped
            act!(env, 10.0f0)  # Should be clamped to 1.0
            @test env.problem.force == 1.0f0

            reset!(env)
            act!(env, -10.0f0)  # Should be clamped to -1.0
            @test env.problem.force == -1.0f0
        end

        @testset "Array action input" begin
            reset!(env)
            act!(env, [0.5f0])  # Should work with array input
            @test env.problem.force == 0.5f0
        end
    end

    @testset "Common observation space" begin
        discrete_env = MountainCarEnv()
        continuous_env = MountainCarContinuousEnv()

        # Both should have identical observation spaces
        os_discrete = observation_space(discrete_env)
        os_continuous = observation_space(continuous_env)

        @test os_discrete.low == os_continuous.low
        @test os_discrete.high == os_continuous.high
        @test os_discrete.shape == os_continuous.shape

        @test os_discrete.low[1] == discrete_env.problem.min_position
        @test os_discrete.high[1] == discrete_env.problem.max_position
        @test os_discrete.low[2] == -discrete_env.problem.max_speed
        @test os_discrete.high[2] == discrete_env.problem.max_speed
    end

    @testset "AbstractMountainCarEnv type hierarchy" begin
        discrete_env = MountainCarEnv()
        continuous_env = MountainCarContinuousEnv()

        @test discrete_env isa AbstractMountainCarEnv
        @test continuous_env isa AbstractMountainCarEnv
        @test discrete_env isa AbstractEnv
        @test continuous_env isa AbstractEnv
    end
end