# Taxi-v3-Reinforcement-learning
### Features
-The Generlizae Reinforcment Learning Project
-using Grid search 
### Set_up_environment 
	!pip install cmake 'gym[atari]' scipy
### Import library
	import gym
	from IPython.display import clear_output
	from time import sleep
	import random
	from IPython.display import clear_output
	import numpy as np
	from IPython.display import clear_output
	from time import sleep
###  Functions
### Define Environment 
	def env_def(env_name):
  	env=gym.make(env_name).env
	  env.reset() # reset environment to a new, random state
 	 env.render()
  	print("Action Space {}".format(env.action_space))
 	 print("State Space {}".format(env.observation_space))
 	 return env
### define brute force to choose random 
	def BrouteForce(env):
 	 env.s = 328  # set environment to illustration's state

 	 epochs = 0
  	penalties, rewards = 0, 0

  	frames = [] # for animation

  	done = False

  	while not done:
      action = env.action_space.sample()
      state, reward, done, info = env.step(action)

      if reward == -10:
          penalties += 1
      
      if reward > 0:
        rewards += 1

      # Put each rendered frame into dict for animation
      frames.append({
          'frame': env.render(mode='ansi'),
          'state': state,
          'action': action,
          'reward': reward
          }
      )

      epochs += 1
      
      
  	print("Timesteps taken: {}".format(epochs))
  	print("Penalties incurred: {}".format(penalties))
  	return frames
###  frames for visualization 
	def Frames_pt(frames):
 	 for i, frame in enumerate(frames):
      clear_output(wait=True)
      #print(frame['frame'].getvalue())
      print(frame['frame'])
      print(f"Timestep: {i + 1}")
      print(f"State: {frame['state']}")
      print(f"Action: {frame['action']}")
      print(f"Reward: {frame['reward']}")
      sleep(.1)
### Define q table and  hyperparameters and decay episoln to explore at the beginning and then  explorate 
	def Qlearning_DECAY(alpha,gamma,epsilon,env):
	  q_table = np.zeros([env.observation_space.n, env.action_space.n]) #Initialize the q table
  # For plotting metrics
 	 all_epochs = []
  	all_penalties = []
 	 decayrate=0.1
 	 for i in range(1, 100001):
      state = env.reset()

      epochs, penalties, reward, = 0, 0, 0
      done = False
      if i%10000:
        alpha = abs(alpha - (1/(1 + (decayrate * 100000))) * alpha)
        gamma = abs(gamma - (1/(1 + (decayrate * 100000))) * gamma)
        epsilon = abs(alpha - (1/(1 + (decayrate * 100000))) * epsilon)
      while not done:
          if random.uniform(0, 1) < epsilon:
              action = env.action_space.sample() # Explore action space
          else:
              action = np.argmax(q_table[state]) # Exploit learned values

          next_state, reward, done, info = env.step(action) 
          
          old_value = q_table[state, action]
          next_max = np.max(q_table[next_state])
      
          new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
          q_table[state, action] = new_value

          if reward == -10:
              penalties += 1

          state = next_state
          epochs += 1
          
      if i % 100 == 0:
          clear_output(wait=True)
          print(f"Episode: {i}")

  
  	print("Training finished.\n")
	  return q_table
### evaluation
	def evaluation(q_table,env):
 	 total_epochs, total_penalties = 0, 0
 	 episodes = 1000
 	 total_reward=0
	  for _ in range(episodes):
      state = env.reset()
      epochs, penalties, reward = 0, 0, 0
      
      done = False
      
      while not done:
          action = np.argmax(q_table[state])
          state, reward, done, info = env.step(action)

          if reward == -10:
              penalties += 1

          epochs += 1

      total_penalties += penalties
      total_reward+= reward
      total_epochs += epochs

 	 print(f"Results after {episodes} episodes:")
 	 print(f"Average timesteps per episode: {total_epochs / episodes}")
 	 print(f"Average penalties per episode: {total_penalties / episodes}")
 	 print(f"Average penalties per episode: {total_reward / episodes}")
	  return total_reward
### Grid search  to choose the best hyperparameter
	def grid_search(env):
 	 alpha_range = list(np.arange(0, 1, 0.5))
  	gama_range = list(np.arange(0, 1, 0.5))
  	epsilon_range = list(np.arange(0, 1, 0.5))
 	 Max=0
 	 best_Alpha=0
  	best_Gamma=0
 	 best_Epsilon=0
  	for al in alpha_range:
    for gama in gama_range:
      for ep in epsilon_range:
        q_table=Qlearning_DECAY(al,gama,ep,env)
        total_reward=evaluation(q_table,env)
        if total_reward>Max:
          Max=total_reward
          best_Alpha=al
          best_Gamma=gama
          best_Epsilon=ep

 	 return best_Alpha,best_Gamma, best_Epsilon
## Main
	env=env_def ("Taxi-v3")
	frames = BrouteForce(env)
	Frames_pt
	q = Qlearning_DECAY(0.8,0.8,0.7,env)
	evaluation(q)
	best_Alpha,best_Gamma, best_Epsilon=grid_search(env)

`

###End
