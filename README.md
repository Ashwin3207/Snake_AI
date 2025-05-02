# 🐍 Snake Game AI with Deep Q-Learning (DQN)

An AI-powered Snake game built with **Deep Q-Learning (DQN)** using **PyTorch**. The agent learns to survive, eat food, and avoid self-collision and walls by interacting with a custom game environment.

---

## 📌 Features

- 🧠 Deep Q-Learning with experience replay and epsilon-greedy strategy
- 🎮 Fully custom Snake environment built from scratch
- 📊 Real-time visualization of training progress
- 💾 Model saving and testing
- 🧱 Modular codebase (easy to extend or experiment)

---

## 🗂️ Project Structure

```
snake-dqn/
├── dqn_agent.py       # DQN model and logic for the agent
├── snake_env.py       # Custom Snake environment (gym-like)
├── train.py           # Training loop and visualization
├── main.py            # Test/play the trained AI agent
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/snake-dqn.git
cd snake-dqn
```

### 2. Install Dependencies
Make sure you have Python 3.7 or higher installed.

```bash
pip install -r requirements.txt
```

### 3. (Optional) Create and Activate a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

---

## 🧪 How to Run the Project

### 🎓 Train the AI Agent
```bash
python train.py
```
- Starts training using Deep Q-Learning
- Saves model as `model.pth` after training
- Visualizes rewards and scores if plotting is enabled

### 🕹️ Test the Trained Agent
```bash
python main.py
```
- Loads the trained model (`model.pth`)
- Plays the Snake game using the AI
- View the gameplay in a Pygame window

---

## 🛠️ Hyperparameters (Defined in `train.py`)

| Parameter         | Description                                       |
|-------------------|---------------------------------------------------|
| `gamma`           | Discount factor for future rewards                |
| `epsilon`         | Exploration rate (decreases over time)            |
| `lr`              | Learning rate for optimizer                       |
| `batch_size`      | Number of experiences per training step           |
| `max_memory`      | Replay buffer size                                |
| `episodes`        | Number of episodes to train the agent             |

Feel free to tweak these to improve training results.

---

## 🎮 Game Rules

- The snake starts at length 1 and moves automatically
- Eating food increases the snake’s length and score
- Colliding with a wall or itself ends the game
- Rewards:
  - `+10` for eating food
  - `-10` for dying
  - `-0.1` for each step (to encourage faster food seeking)

---

## 📊 Visualization

You can enable reward/scores plotting using `matplotlib` inside `train.py`.  
This will help visualize the AI's progress over episodes.

---

## 📦 Requirements

All required packages are listed in `requirements.txt`. Install using:
```bash
pip install -r requirements.txt
```

Sample content:
```
torch
numpy
matplotlib
pygame
```

---

## 🧠 Tech Stack

- **Python 3.7+**
- **PyTorch** - neural network and Q-learning
- **NumPy** - numerical operations
- **Matplotlib** - training graph plotting
- **Pygame** - rendering the Snake game

---

## ✅ Future Improvements

- Use Double DQN or Dueling DQN
- Add CNN-based state representation
- Save training history and metrics
- Web-based visualization of gameplay

---

## 📄 License

This project is licensed under the **MIT License**.  
You're free to use, modify, and distribute it.

---

## 🙌 Acknowledgments

- Inspired by DeepMind's DQN paper
- Thanks to Python, PyTorch, and Pygame communities
- Made with ❤️ by Ashwin
