# Snake Game AI with Deep Q-Learning (DQN)

This project is an implementation of a Snake game AI using **Deep Q-Learning** (DQN) in **PyTorch**. The AI learns to control the Snake by interacting with the game environment, aiming to eat food, avoid boundaries, prevent self-collisions, and optimize its strategy to maximize its score. The project includes a custom game environment (`SnakeEnv`), where the agent explores different actions and learns through reinforcement learning.

## Features

- **Deep Q-Learning (DQN)**: The AI agent learns by exploring the environment, updating Q-values using a neural network.
- **Custom Game Environment**: The environment simulates the classic Snake game, incorporating game mechanics such as food spawning, movement, boundary checks, and collision detection.
- **Training Loop**: The training process involves experience collection, Q-network updates, and an epsilon-greedy policy for exploration.
- **Visualization**: Training progress and AI performance are visualized, showing how the AI improves over time.

## Project Structure

- **`dqn_agent.py`**: Contains the implementation of the Deep Q-Learning agent, which interacts with the environment and learns from experiences.
- **`snake_env.py`**: Defines the Snake game environment, including the rules of the game, state space, reward system, and actions.
- **`main.py`**: The entry point for running the game and testing the AI. It initializes the environment and the agent and allows you to see how the AI performs in the game.
- **`train.py`**: Script for training the agent, including the setup of hyperparameters, the training loop, and model saving. This is where the main learning process takes place.

## Technologies Used

- **Python**: The main programming language for the project.
- **PyTorch**: Used for building and training the neural network for Q-learning.
- **NumPy**: Utilized for handling arrays and matrix operations.
- **Matplotlib** (optional for visualization): Used for visualizing the training process.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/snake-game-ai.git
   cd snake-game-ai
Install the required dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Ensure you have Python 3.x and pip installed. You can check this by running:

bash
Copy
Edit
python --version
pip --version
(Optional) Set up a virtual environment for isolated dependency management:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
How to Run
Train the AI: To train the AI using Deep Q-Learning, run:

bash
Copy
Edit
python train.py
This will begin the training process and save the model after training.

Test the AI: Once the model is trained, you can test it by running:

bash
Copy
Edit
python main.py
This will start the game with the trained AI controlling the Snake.

Visualize the Training Process: During the training process, you can modify the script to visualize the performance of the AI. You can use libraries like matplotlib to plot rewards over time or visualize the agent's actions in real-time.

Hyperparameters
The following hyperparameters are used in the training process:

Learning Rate: The step size for updating Q-values.

Gamma (Discount Factor): The discount factor used to calculate the discounted future reward.

Epsilon (Exploration Rate): The probability of selecting a random action during training (epsilon-greedy policy).

Batch Size: The number of experiences sampled from the replay buffer during each training step.

Replay Buffer Size: The maximum size of the replay buffer, which stores past experiences for training.

These parameters can be adjusted in the train.py script.

Game Rules
The Snake starts with a length of 1 and grows each time it eats food.

The goal is to eat food while avoiding collisions with the walls and the Snake's own body.

The game ends when the Snake collides with itself or the boundaries.
