using ClassicControlEnvironments
using DRiL
using WGLMakie
## setup env, alg, policy and agent
alg = PPO()
cartpoleenv = BroadcastedParallelEnv([CartPoleEnv() for _ in 1:8])
cartpoleenv = MonitorWrapperEnv(cartpoleenv)
cartpoleenv = NormalizeWrapperEnv(cartpoleenv, gamma=alg.gamma)

cartpolepolicy = ActorCriticPolicy(observation_space(cartpoleenv), action_space(cartpoleenv))
cartpoleagent = ActorCriticAgent(cartpolepolicy; verbose=2, n_steps=128, batch_size=128, learning_rate=3f-4, epochs=10)
## train agent
learn_stats = learn!(cartpoleagent, cartpoleenv, alg; max_steps=100_000)
## collect trajectory
single_env = CartPoleEnv()
obs, actions, rewards = collect_trajectory(cartpoleagent, single_env; norm_env=cartpoleenv)
sum(rewards)
## plot trajectory
fig_traj = plot_trajectory(single_env, obs, actions, rewards)
plot_trajectory_interactive(single_env, obs, actions, rewards)