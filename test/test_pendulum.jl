using ClassicControlEnvironments
using DRiL
using Test
using Random

@testset "PendulumEnv tests" begin

    @testset "Observation range" begin
        env = PendulumEnv()

        # Test observations after initialization
        obs = observe(env)
        @test length(obs) == 3
        @test all(-1.0f0 .<= obs .<= 1.0f0)

        # Test observations after random actions
        rng = MersenneTwister(123)
        for _ in 1:100
            action = rand(rng, Float32) * 4.0f0 - 2.0f0  # Random action in [-2, 2]
            act!(env, action)
            obs = observe(env)
            @test all(-1.0f0 .<= obs .<= 1.0f0)
            @test -1.0f0 <= obs[1] <= 1.0f0  # cos(theta)
            @test -1.0f0 <= obs[2] <= 1.0f0  # sin(theta)
            @test -1.0f0 <= obs[3] <= 1.0f0  # scaled velocity
        end
    end

    @testset "Zero action and gravity" begin
        # Create environment with zero gravity
        env = PendulumEnv(gravity=0.0f0)
        reset!(env)

        # Initial angle and velocity
        init_theta = env.problem.theta
        init_vel = env.problem.velocity

        # Take steps with zero action
        for _ in 1:100
            act!(env, 0.0f0)

            # Assert theta remains within [-π, π]
            @test -π <= env.problem.theta <= π

            # With zero gravity and zero action, velocity should remain constant
            # and theta should change linearly
            @test isapprox(env.problem.velocity, init_vel, atol=1e-5)
            if abs(init_vel) < 1e-5
                @test isapprox(env.problem.theta, init_theta, atol=1e-5)
            end
        end
    end

    @testset "Reward bounds" begin
        env = PendulumEnv()

        # Theoretical reward bounds
        min_reward = -(π^2 + 0.1f0 * 8.0f0^2 + 0.001f0 * 2.0f0^2)
        max_reward = 0.0f0

        # Test rewards for random states
        rng = MersenneTwister(456)
        for _ in 1:100
            reset!(env)

            # Random action
            action = rand(rng, Float32) * 4.0f0 - 2.0f0
            current_reward = act!(env, action)

            # Check reward bounds
            @test min_reward <= current_reward <= max_reward

            # Check each component of the reward
            theta_penalty, vel_penalty, action_penalty = Pendulum.pendulum_rewards(
                env.problem.theta, env.problem.velocity, action)

            @test theta_penalty <= 0.0f0
            @test vel_penalty <= 0.0f0
            @test action_penalty <= 0.0f0

            # Check that sum matches overall reward
            @test isapprox(current_reward, theta_penalty + vel_penalty + action_penalty, atol=1e-5)
        end
    end

    @testset "Step count and truncation" begin
        max_steps = 50
        env = PendulumEnv(max_steps=max_steps)
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
end