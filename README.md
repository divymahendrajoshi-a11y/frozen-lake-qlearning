# Frozen Lake Q-Learning Agent

A Reinforcement Learning agent that learns to solve the **Frozen Lake** environment using the **Q-Learning** algorithm.

## 📌 About the Project
This project implements a Q-Learning agent from scratch to navigate the Frozen Lake environment (from OpenAI Gym / Gymnasium). The agent learns an optimal policy through trial and error by updating a Q-table based on rewards received from the environment.

## 📂 Project Structurefrozen-lake-qlearning/
│
├── train.py        # Trains the Q-Learning agent and saves the Q-table
├── test.py         # Loads the trained Q-table and evaluates the agent
├── q_table.npy      # Saved Q-table (trained model)
└── .gitignore       # Files/folders excluded from version control

## ⚙️ How It Works
1. The agent starts with no knowledge of the environment
2. It explores the environment and updates the Q-table using the Q-Learning update rule
3. Over multiple episodes, the agent learns the best actions to take in each state
4. The trained Q-table is saved and later used for testing

## 🚀 Getting Started

### Prerequisites
```bash
pip install gymnasium numpy
```

### Training the Agent
```bash
python train.py
```

### Testing the Agent
```bash
python test.py
```

## 📊 Results
The trained agent successfully learns to reach the goal state while avoiding holes in the Frozen Lake environment.

## 🛠️ Built With
- Python
- NumPy
- Gymnasium (OpenAI Gym)

## 📄 License
This project is open source and available for learning purposes.
