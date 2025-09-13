import random
import numpy as np
import pygame
from collections import deque

class SnakeEnv:
    def __init__(self, width=12, height=12, render=False):
        self.width = width
        self.height = height
        self.render_mode = render
        self.cell_size = 30        
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((width * self.cell_size, height * self.cell_size))
            pygame.display.set_caption('Snake RL')
            self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        # Start snake in center
        center_x, center_y = self.width // 2, self.height // 2
        self.snake = deque([[center_x, center_y]])
        self.direction = 'RIGHT'
        self.place_food()
        self.score = 0
        return self.get_state()
    
    def place_food(self):
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if [x, y] not in self.snake:
                self.food = [x, y]
                break
    
    def step(self, action):
        old_head = self.snake[0]
        old_dist = abs(old_head[0] - self.food[0]) + abs(old_head[1] - self.food[1])
        self.update_direction(action)
        self.move_snake()
        head = self.snake[0]
        new_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        reward = 0.0
        done = False
        # Check collisions first (terminal states)
        if (head[0] < 0 or head[0] >= self.width or 
            head[1] < 0 or head[1] >= self.height):
            reward = -2.0
            done = True
            return self.get_state(), reward, done
        if head in list(self.snake)[1:]:
            reward = -2.0
            done = True
            return self.get_state(), reward, done
        # Food eating reward
        if head == self.food:
            reward += 10.0
            self.score += 1
            self.place_food()
            # Additional reward based on snake length
            reward += min(0.5, len(self.snake) * 0.05)
        else:
            self.snake.pop()
            # Distance-based reward with more meaningful scaling
            dist_improvement = (old_dist - new_dist) / (self.width + self.height)
            reward += dist_improvement * 3.0
        # Survival bonus that increases with score
        reward += 0.001 * (1 + self.score * 0.1)
        # Bonus reward for high scores
        if self.score >= 20:
            reward += 0.1  # Add bonus for scores above 20
        elif self.score >= 30:
            reward += 0.2  # Add more bonus for scores above 30
        return self.get_state(), reward, done
    
    def is_food_reachable(self, head):
        """Flood-fill from the snake's head to see if the bait is in the reachable zone."""
        visited = set(map(tuple, self.snake))  # body is an obstacle
        q = deque([tuple(head)])
        steps = 0
        while q:
            x, y = q.popleft()
            steps += 1
            if (x, y) == tuple(self.food):
                return True
            for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        q.append((nx, ny))
        return False
    
    def update_direction(self, action):
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        current_idx = directions.index(self.direction)
        if action == 1:  # Right turn
            new_idx = (current_idx + 1) % 4
        elif action == 2:  # Left turn
            new_idx = (current_idx - 1) % 4
        else:  # Straight
            new_idx = current_idx
        self.direction = directions[new_idx]
    
    def move_snake(self):
        head = self.snake[0].copy()
        if self.direction == 'UP':
            head[1] -= 1
        elif self.direction == 'DOWN':
            head[1] += 1
        elif self.direction == 'LEFT':
            head[0] -= 1
        elif self.direction == 'RIGHT':
            head[0] += 1
        self.snake.appendleft(head)
    
    def get_state(self):
        head = self.snake[0]
        # Danger detection (wall or body collision)
        danger_straight = self.is_danger(0)
        danger_right = self.is_danger(1)
        danger_left = self.is_danger(2)
        # Direction one-hot encoding
        dir_up = 1 if self.direction == 'UP' else 0
        dir_right = 1 if self.direction == 'RIGHT' else 0
        dir_down = 1 if self.direction == 'DOWN' else 0
        dir_left = 1 if self.direction == 'LEFT' else 0
        # Food direction relative to head
        food_up = 1 if self.food[1] < head[1] else 0
        food_right = 1 if self.food[0] > head[0] else 0
        food_down = 1 if self.food[1] > head[1] else 0
        food_left = 1 if self.food[0] < head[0] else 0
        # Distance to walls (normalized)
        wall_left = head[0] / self.width
        wall_right = (self.width - head[0] - 1) / self.width
        wall_up = head[1] / self.height
        wall_down = (self.height - head[1] - 1) / self.height
        # Snake length normalized
        snake_length = len(self.snake) / (self.width * self.height)
        # Food distance normalized
        food_dist_x = (self.food[0] - head[0]) / self.width
        food_dist_y = (self.food[1] - head[1]) / self.height
        state = np.array([
            danger_left, danger_straight, danger_right,  # 3 danger signals
            dir_left, dir_up, dir_right, dir_down,       # 4 direction
            food_left, food_up, food_right, food_down,   # 4 food direction
            wall_left, wall_right, wall_up, wall_down,   # 4 wall distances
            snake_length,                                # snake length
            food_dist_x, food_dist_y                     # 2 food distance
        ], dtype=np.float32)
        return state
    
    def is_danger(self, action):
        # Get next head position for given action
        head = self.snake[0].copy()
        # Apply action to get new direction
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        current_idx = directions.index(self.direction)
        if action == 1:  # Right turn
            new_idx = (current_idx + 1) % 4
        elif action == 2:  # Left turn
            new_idx = (current_idx - 1) % 4
        else:  # Straight
            new_idx = current_idx
        new_direction = directions[new_idx]
        # Move head in new direction
        if new_direction == 'UP':
            head[1] -= 1
        elif new_direction == 'DOWN':
            head[1] += 1
        elif new_direction == 'LEFT':
            head[0] -= 1
        elif new_direction == 'RIGHT':
            head[0] += 1
        # Check if new position is dangerous
        if (head[0] < 0 or head[0] >= self.width or 
            head[1] < 0 or head[1] >= self.height or 
            head in list(self.snake)):
            return 1
        return 0
    
    def render(self):
        if not self.render_mode:
            return
        # Colors
        BLACK = (0, 0, 0)
        GREEN = (0, 255, 0)
        RED = (255, 0, 0)
        WHITE = (255, 255, 255)
        self.screen.fill(BLACK)
        # Draw snake
        for i, segment in enumerate(self.snake):
            color = GREEN if i == 0 else (0, 200, 0)  # Head brighter
            rect = (segment[0] * self.cell_size, segment[1] * self.cell_size, 
                   self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, color, rect)
        # Draw food
        food_rect = (self.food[0] * self.cell_size, self.food[1] * self.cell_size, 
                    self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, RED, food_rect)
        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f'Score: {self.score}', True, WHITE)
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()
        self.clock.tick(10)  # 10 FPS
    
    def close(self):
        if self.render_mode:
            pygame.quit()