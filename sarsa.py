import numpy as np
from tqdm import tqdm
import gym
import gym_gridworlds

env = gym.make('WindyGridworld-v0')

def create_policy(Q, num_actions, epsilon):
	def act(state):
		action_probas = np.ones(num_actions, dtype=float) * epsilon/num_actions
		best_action = np.argmax(Q[state])
		action_probas[best_action] += (1 - epsilon)
		return action_probas
	return act

def SARSAlambda_learner(env, num_episodes, epsilon=0.1, alpha=0.5, lambdaa=0.9, discount_factor=0.9):
	Q = np.zeros((env.observation_space[0].n, env.observation_space[1].n, env.action_space.n))
	n_steps = []
	for ith_episode in tqdm(range(num_episodes)):
		state = env.reset()
		count_states = 0
		while True:
			policy = create_policy(Q, env.action_space.n, epsilon=1/(ith_episode+1))
			action = np.random.choice(4, p=policy(state))
			next_state, reward, done, _ = env.step(action)
			next_action = np.random.choice(4, p=policy(next_state))
			td_error = reward + (discount_factor * Q[next_state[0], next_state[1], next_action] - Q[state[0], state[1], action])
			Q[state[0], state[1], action] += (alpha * td_error)
			# print("Action Taken: ", action)
			# print(done)
			# print(next_state)
			count_states += 1
			if done==True:
				break
			state = next_state
		# print(count_states, ith_episode)
		n_steps.append(count_states)
		# print(Q)
		# print("-x-EOE-x-")
	return Q, n_steps

Q, n_steps = SARSAlambda_learner(env, 2000)
print(n_steps)
print(np.min(np.array(n_steps)))
# Q = np.zeros((env.observation_space[0].n, env.observation_space[1].n, env.action_space.n))
# policy = create_policy(Q, env.action_space.n, 0.8)
# print(policy(env.reset()))
# print(np.random.choice(4, p=policy(env.reset())))
