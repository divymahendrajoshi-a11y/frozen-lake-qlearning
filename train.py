import gymnasium as gym
import numpy as np

# Environment banao
env = gym.make('FrozenLake-v1', is_slippery=False)

# Q-table banao - 16 states x 4 actions, sab zero se shuru
q_table = np.zeros((env.observation_space.n, env.action_space.n))

# Hyperparameters (ye values control karte hain agent kaise seekhega)
alpha = 0.8         # learning rate - kitna fast seekhe
gamma = 0.95        # discount factor - future reward ki value
epsilon = 1.0        # exploration rate - shuru mein zyada random try karega
epsilon_decay = 0.001
episodes = 5000      # kitni baar agent practice karega

# Training loop
for episode in range(episodes):
    state, info = env.reset()
    done = False

    while not done:
        # Epsilon-greedy: kabhi random action (explore), kabhi best known action (exploit)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()   # random action
        else:
            action = np.argmax(q_table[state])   # best known action

        # Action lo, next state aur reward pao
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Q-table update karo (yehi hai Q-learning ka core formula)
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[new_state]) - q_table[state, action]
        )

        state = new_state

    # Epsilon ko dheere-dheere kam karo (jitna train hota jaye utna kam random)
    epsilon = max(0.01, epsilon - epsilon_decay)

    # Har 500 episodes pe progress dikhao
    if (episode + 1) % 500 == 0:
        print(f"Episode {episode + 1}/{episodes} complete")

# Trained Q-table save karo
np.save('q_table.npy', q_table)
print("\nTraining complete!")
print("\nFinal Q-table:")
print(q_table)