import random
import numpy as np
import pygame
from typing import Tuple, List, Optional, Dict, Any

# Constants
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
BLOCK_SIZE = 20
GRID_WIDTH = SCREEN_WIDTH // BLOCK_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // BLOCK_SIZE

# Colors
COLORS = {
    'WHITE': (255, 255, 255),
    'GREEN': (0, 255, 0),
    'RED': (255, 0, 0),
    'BLUE': (0, 0, 255),
    'BLACK': (0, 0, 0),
    'HEAD_COLOR': (0, 200, 0)  # Different color for head
}

class SnakeEnv:
    def __init__(self, render_mode: Optional[str] = None):
        self.render_mode = render_mode
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption('Snake Game')
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont('Arial', 20)
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.snake = [(5 * BLOCK_SIZE, 5 * BLOCK_SIZE), 
                     (4 * BLOCK_SIZE, 5 * BLOCK_SIZE), 
                     (3 * BLOCK_SIZE, 5 * BLOCK_SIZE)]  # Center position
        self.direction = (BLOCK_SIZE, 0)  # Initial direction (right)
        self.food = self._place_food()
        self.done = False
        self.score = 0
        self.steps = 0
        self.last_distance_to_food = self._manhattan_distance(self.snake[0], self.food)
        self.visited_positions = set(self.snake)
        
        if self.render_mode == "human":
            self._render_frame()
            
        return self.get_state()

    def _place_food(self) -> Tuple[int, int]:
        """Place food at random position not occupied by snake."""
        available_positions = set(
            (x * BLOCK_SIZE, y * BLOCK_SIZE)
            for x in range(GRID_WIDTH)
            for y in range(GRID_HEIGHT)
        ) - set(self.snake)
        
        if not available_positions:
            return (-1, -1)  # No available positions
        
        return random.choice(list(available_positions))

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_state(self) -> np.ndarray:
        """Get current state representation."""
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        direction_x, direction_y = self.direction

        # Normalized state features
        state = [
            # Normalized positions
            head_x / SCREEN_WIDTH,
            head_y / SCREEN_HEIGHT,
            food_x / SCREEN_WIDTH,
            food_y / SCREEN_HEIGHT,
            
            # Direction information
            direction_x / BLOCK_SIZE,
            direction_y / BLOCK_SIZE,
            
            # One-hot direction encoding
            int(self.direction == (BLOCK_SIZE, 0)),   # Right
            int(self.direction == (-BLOCK_SIZE, 0)),  # Left
            int(self.direction == (0, -BLOCK_SIZE)),  # Up
            int(self.direction == (0, BLOCK_SIZE)),    # Down
            
            # Danger detection
            int((head_x + BLOCK_SIZE) >= SCREEN_WIDTH),  # Danger right
            int((head_x - BLOCK_SIZE) < 0),             # Danger left
            int((head_y - BLOCK_SIZE) < 0),              # Danger up
            int((head_y + BLOCK_SIZE) >= SCREEN_HEIGHT), # Danger down
            
            # Body collision detection
            int((head_x + BLOCK_SIZE, head_y) in self.visited_positions),  # Right
            int((head_x - BLOCK_SIZE, head_y) in self.visited_positions),  # Left
            int((head_x, head_y - BLOCK_SIZE) in self.visited_positions),  # Up
            int((head_x, head_y + BLOCK_SIZE) in self.visited_positions),  # Down
            
            # Score and steps (normalized)
            self.score / 100.0,
            self.steps / 1000.0
        ]

        return np.array(state, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one time step in the environment."""
        self.steps += 1

        # Validate action and change direction
        if action == 0 and self.direction != (BLOCK_SIZE, 0):  # Left
            self.direction = (-BLOCK_SIZE, 0)
        elif action == 1 and self.direction != (-BLOCK_SIZE, 0):  # Right
            self.direction = (BLOCK_SIZE, 0)
        elif action == 2 and self.direction != (0, BLOCK_SIZE):  # Up
            self.direction = (0, -BLOCK_SIZE)
        elif action == 3 and self.direction != (0, -BLOCK_SIZE):  # Down
            self.direction = (0, BLOCK_SIZE)

        # Move snake
        head_x, head_y = self.snake[0]
        new_head = (head_x + self.direction[0], head_y + self.direction[1])

        self.snake.insert(0, new_head)
        self.visited_positions.add(new_head)

        # Reward for moving closer to food
        reward = -0.1  # Small penalty for each step

        # Check for food consumption
        if new_head == self.food:
            self.score += 1
            reward = 10.0
            self.food = self._place_food()
            if self.food == (-1, -1):  # Board full
                self.done = True
                reward = 100.0
        else:
            tail = self.snake.pop()
            self.visited_positions.remove(tail)

        # Check for collisions
        if (new_head[0] < 0 or new_head[0] >= SCREEN_WIDTH or 
            new_head[1] < 0 or new_head[1] >= SCREEN_HEIGHT or 
            new_head in self.snake[1:]):  # Snake collides with itself or boundary
            self.done = True
            reward = -10.0

        # Distance-based reward (moving closer to food)
        current_distance = self._manhattan_distance(new_head, self.food)
        if current_distance < self.last_distance_to_food:
            reward += 1.0  # Reward for moving closer
        else:
            reward -= 0.5  # Penalty for moving away
        self.last_distance_to_food = current_distance

        # Render the frame
        if self.render_mode == "human":
            self._render_frame()

        return self.get_state(), reward, self.done, {"score": self.score, "steps": self.steps}

    def _render_frame(self) -> None:
        """Render one frame of the environment."""
        self.screen.fill(COLORS['BLACK'])

        # Draw food
        pygame.draw.rect(self.screen, COLORS['RED'], pygame.Rect(self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))

        # Draw snake (head different color)
        for i, segment in enumerate(self.snake):
            color = COLORS['HEAD_COLOR'] if i == 0 else COLORS['GREEN']
            pygame.draw.rect(self.screen, color, pygame.Rect(segment[0], segment[1], BLOCK_SIZE, BLOCK_SIZE))

        # Draw score
        score_text = self.font.render(f'Score: {self.score}', True, COLORS['WHITE'])
        self.screen.blit(score_text, (5, 5))

        pygame.display.flip()
        self.clock.tick(15)  # Control game speed

    def render(self):
        # This function can call the existing pygame drawing code
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self) -> None:
        """Close the environment and release resources."""
        if self.render_mode == "human":
            pygame.quit()
