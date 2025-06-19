using ClassicControlEnvironments
using DRiL
using WGLMakie
## setup env, alg, policy and agent
alg = PPO(; ent_coef=0f0, vf_coef=0.480177f0, gamma=0.990886f0, gae_lambda=0.85821f0, clip_range=0.132141f0)
pendenv = BroadcastedParallelEnv([PendulumEnv() for _ in 1:8])
pendenv = MonitorWrapperEnv(pendenv)
pendenv = NormalizeWrapperEnv(pendenv, gamma=alg.gamma)

pendpolicy = ActorCriticPolicy(observation_space(pendenv), action_space(pendenv))
pendagent = ActorCriticAgent(pendpolicy; verbose=2, n_steps=128, batch_size=128, learning_rate=1.95409f-4, epochs=20)
## train agent
learn_stats = learn!(pendagent, pendenv, alg; max_steps=100_000)
## collect trajectory
single_env = PendulumEnv()
obs, actions, rewards = collect_trajectory(pendagent, single_env; norm_env=pendenv)
sum(rewards)
## plot trajectory
fig_traj = plot_trajectory(single_env, obs, actions, rewards)
plot_trajectory_interactive(single_env, obs, actions, rewards)