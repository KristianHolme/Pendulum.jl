using ClassicControlEnvironments
using DRiL
using WGLMakie
using Zygote
## setup env, alg, policy and agent
alg = PPO()
acrobotenv = BroadcastedParallelEnv([AcrobotEnv() for _ in 1:8])
acrobotenv = MonitorWrapperEnv(acrobotenv)
acrobotenv = NormalizeWrapperEnv(acrobotenv, gamma=alg.gamma)

acrobotpolicy = ActorCriticPolicy(observation_space(acrobotenv), action_space(acrobotenv))
acrobotagent = ActorCriticAgent(acrobotpolicy; verbose=2, n_steps=128, batch_size=128, learning_rate=3f-4, epochs=10)
## train agent
learn_stats = learn!(acrobotagent, acrobotenv, alg; max_steps=100_000)
## collect trajectory
single_env = AcrobotEnv()
obs, actions, rewards = collect_trajectory(acrobotagent, single_env; norm_env=acrobotenv)
sum(rewards)
## plot trajectory
fig_traj = plot_trajectory(single_env, obs, actions, rewards)
plot_trajectory_interactive(single_env, obs, actions, rewards)
