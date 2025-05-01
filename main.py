import numpy as np
import random
import pygame
from snake_env import SnakeEnv

# Constants
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
BLOCK_SIZE = 20

class SafeFoodChasingAgent:
    def __init__(self):
        self.actions = [0, 1, 2, 3]  # Left, Right, Up, Down

    def select_action(self, state, snake_head, food_pos, snake_body):
        """Select an action based on the relative position of the food and avoiding collisions."""
        head_x, head_y = snake_head
        food_x, food_y = food_pos

        # Calculate relative direction (dx, dy)
        dx = food_x - head_x
        dy = food_y - head_y

        # Potential actions (left, right, up, down)
        possible_actions = []

        # Avoid moving out of bounds or into the body
        if head_x + BLOCK_SIZE < SCREEN_WIDTH and (head_x + BLOCK_SIZE, head_y) not in snake_body:  # Right
            possible_actions.append(1)
        if head_x - BLOCK_SIZE >= 0 and (head_x - BLOCK_SIZE, head_y) not in snake_body:  # Left
            possible_actions.append(0)
        if head_y - BLOCK_SIZE >= 0 and (head_x, head_y - BLOCK_SIZE) not in snake_body:  # Up
            possible_actions.append(2)
        if head_y + BLOCK_SIZE < SCREEN_HEIGHT and (head_x, head_y + BLOCK_SIZE) not in snake_body:  # Down
            possible_actions.append(3)

        # If no valid move (rare but can happen if the snake is surrounded), move randomly
        if not possible_actions:
            possible_actions = self.actions

        # Prioritize moving towards the food, but avoid collisions
        if abs(dx) > abs(dy):  # Horizontal priority
            if dx > 0 and 1 in possible_actions:  # Move right
                return 1
            elif dx < 0 and 0 in possible_actions:  # Move left
                return 0
        else:  # Vertical priority
            if dy > 0 and 3 in possible_actions:  # Move down
                return 3
            elif dy < 0 and 2 in possible_actions:  # Move up
                return 2

        # If no direction towards food is valid, pick randomly from possible safe actions
        return random.choice(possible_actions)

def train():
    env = SnakeEnv(render_mode="human")  # or "rgb_array" for AI training
    agent = SafeFoodChasingAgent()
    
    num_games = 1000
    for game in range(num_games):
        print(f"Training game {game + 1}...")
        state = env.reset()
        done = False
        
        while not done:
            # Get current snake head position, body, and food position
            snake_head = env.snake[0]
            snake_body = set(env.snake[1:])  # Exclude the head
            food_pos = env.food
            
            # Choose action based on food position and avoiding collisions
            action = agent.select_action(state, snake_head, food_pos, snake_body)
            next_state, reward, done, info = env.step(action)
            state = next_state  # Update state

        print(f"Game {game + 1} | Score: {info['score']}")

        # Reset the game after each game over
        env.reset()

if __name__ == "__main__":
    train()
