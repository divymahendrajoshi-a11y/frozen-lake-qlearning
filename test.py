import gymnasium as gym
import numpy as np
import time

# Trained Q-table load karo
q_table = np.load('q_table.npy')

# Environment banao - is baar "human" render mode taaki dekh sako
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')

episodes_to_test = 5
success_count = 0

for episode in range(episodes_to_test):
    state, info = env.reset()
    done = False
    print(f"\n--- Episode {episode + 1} ---")

    while not done:
        # Ab sirf best action lo (Q-table se), koi random nahi
        action = np.argmax(q_table[state])
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        time.sleep(0.5)   # thoda slow karo taaki dekh sako

    if reward == 1:
        print("Goal tak pahunch gaya! ✅")
        success_count += 1
    else:
        print("Hole mein gir gaya ya fail ho gaya ❌")

print(f"\nSuccess Rate: {success_count}/{episodes_to_test}")
env.close()