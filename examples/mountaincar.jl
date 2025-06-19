using ClassicControlEnvironments
using DRiL
using WGLMakie
## setup env, alg, policy and agent
alg = PPO(;ent_coef=0.01f0)
env = BroadcastedParallelEnv([MountainCarEnv() for _ in 1:16])
env = MonitorWrapperEnv(env)
env = NormalizeWrapperEnv(env, gamma=alg.gamma)

policy = ActorCriticPolicy(observation_space(env), action_space(env))
agent = ActorCriticAgent(policy; verbose=2, n_steps=256, batch_size=64, epochs=10)
## train agent
learn!(agent, env, alg; max_steps=100_000)
## collect trajectory
single_env = MountainCarEnv()
obs, actions, rewards = collect_trajectory(agent, single_env; norm_env=env)
sum(rewards)
## plot trajectory
fig_traj = plot_trajectory(single_env, obs, actions, rewards)
plot_trajectory_interactive(single_env, obs, actions, rewards)