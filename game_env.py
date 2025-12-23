import numpy as np
import random
from collections import deque
import pygame
from numba import njit

@njit(fastmath=True)
def fast_bfs(start_x, start_y, width, height, grid, max_count):
    # 0 = empty, 1 = occupied
    if start_x < 0 or start_x >= width or start_y < 0 or start_y >= height:
        return 0
    if grid[start_x, start_y] == 1:
        return 0
    # Fixed-size queue for speed (simulating deque)
    queue_x = np.empty(width * height, dtype=np.int32)
    queue_y = np.empty(width * height, dtype=np.int32)
    # Visited mask (to avoid modifying the actual grid)
    visited = np.zeros_like(grid)
    # Init 
    head = 0
    tail = 0
    queue_x[tail] = start_x
    queue_y[tail] = start_y
    tail += 1
    visited[start_x, start_y] = 1
    count = 0
    dx = np.array([1, -1, 0, 0])
    dy = np.array([0, 0, 1, -1])
    while head < tail and count < max_count:
        cx = queue_x[head]
        cy = queue_y[head]
        head += 1
        count += 1
        for i in range(4):
            nx = cx + dx[i]
            ny = cy + dy[i]
            if (0 <= nx < width and 0 <= ny < height and 
                grid[nx, ny] == 0 and visited[nx, ny] == 0):
                visited[nx, ny] = 1
                queue_x[tail] = nx
                queue_y[tail] = ny
                tail += 1
    return min(count, max_count)

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
            # Cache static colors and font to avoid reallocations each frame
            self._COLOR_BLACK = (0, 0, 0)
            self._COLOR_GREEN_HEAD = (0, 255, 0)
            self._COLOR_GREEN_BODY = (0, 200, 0)
            self._COLOR_RED = (255, 0, 0)
            self._COLOR_WHITE = (255, 255, 255)
            self._font = pygame.font.Font(None, 36)
        # Pre-compute grid positions for faster food placement
        self.all_positions = [(x, y) for x in range(width) for y in range(height)]
        # Pre-allocate arrays to avoid repeated allocations
        self.state_array = np.zeros(24, dtype=np.float32)
        # Cache for direction calculations
        self.direction_map = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}
        self.directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        self.grid = np.zeros((width, height), dtype=np.int8)
        self.reset()
    
    def reset(self):
        # Start snake in center
        center_x, center_y = self.width // 2, self.height // 2
        self.snake = deque([[center_x, center_y]])
        # Track occupied cells for O(1) collision checks and fast food placement
        self.occupied = {(center_x, center_y)}
        # Reset grid
        self.grid.fill(0)                            
        self.grid[center_x, center_y] = 1
        self.direction = 'RIGHT'
        self.place_food()
        self.score = 0
        self.steps_without_food = 0
        return self.get_state()
    
    def place_food(self):
        # Efficient food placement using set operations with occupied cache
        available_positions = [pos for pos in self.all_positions if pos not in self.occupied]
        if available_positions:
            self.food = list(random.choice(available_positions))
        else:
            # Fallback (shouldn't happen in normal gameplay)
            self.food = [0, 0]

    def get_reachable_spaces(self, pos):
        return fast_bfs(
        int(pos[0]), 
        int(pos[1]), 
        self.width, 
        self.height, 
        self.grid, 
        50
        )
    
    def step(self, action):
        old_head = self.snake[0]
        old_dist = abs(old_head[0] - self.food[0]) + abs(old_head[1] - self.food[1])
        self.steps_without_food += 1
        # Balanced timeout threshold - not too aggressive
        max_steps = self.width * self.height * 3
        if self.steps_without_food > max_steps:
            return self.get_state(), -8.0, True
        self.update_direction(action)
        head = self._compute_next_head()
        # Early wall collision check before mutating structures
        if (head[0] < 0 or head[0] >= self.width or 
            head[1] < 0 or head[1] >= self.height):
            return self.get_state(), -10.0, True
        # Early self-collision using occupied set
        if (head[0], head[1]) in self.occupied:
            return self.get_state(), -10.0, True
        # Commit move
        self.snake.appendleft(head)
        self.occupied.add((head[0], head[1]))
        self.grid[head[0], head[1]] = 1
        new_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        reward = 0.0
        done = False
        # Food consumption - restored more nuanced rewards
        length_ratio = len(self.snake) / (self.width * self.height)
        reward = 0.0
        reward += 0.01 # Survival Reward (Small constant)
        if head == self.food:
            reward += 1.0 + length_ratio
            self.score += 1
            self.steps_without_food = 0
            self.place_food()
        else:
            tail = self.snake.pop()
            self.occupied.discard((tail[0], tail[1]))
            self.grid[tail[0], tail[1]] = 0
            if len(self.snake) < 15:
                if new_dist < old_dist:
                    reward += 0.05
                else:
                    reward -= 0.05
            if len(self.snake) > 10:
                reachable = self.get_reachable_spaces(head)
                if reachable < len(self.snake): # If trapped in space smaller than body
                    reward -= 0.5
        if done:
            penalty = 1.0 + (len(self.snake) / 10.0)
            return self.get_state(), -penalty, True
            
        return self.get_state(), reward, False
    
    def update_direction(self, action):
        current_idx = self.direction_map[self.direction]
        if action == 1:  # Right turn
            new_idx = (current_idx + 1) % 4
        elif action == 2:  # Left turn
            new_idx = (current_idx - 1) % 4
        else:  # Straight
            new_idx = current_idx
        self.direction = self.directions[new_idx]
    
    def _compute_next_head(self):
        head = self.snake[0].copy()
        if self.direction == 'UP':
            head[1] -= 1
        elif self.direction == 'DOWN':
            head[1] += 1
        elif self.direction == 'LEFT':
            head[0] -= 1
        elif self.direction == 'RIGHT':
            head[0] += 1
        return head
    
    def get_state(self):
        """Balanced state computation - keeps important features"""
        head = self.snake[0]
        
        # Basic danger detection
        danger_straight = self.is_danger(0)
        danger_right = self.is_danger(1)
        danger_left = self.is_danger(2)
        
        # Direction encoding
        dir_up = 1 if self.direction == 'UP' else 0
        dir_right = 1 if self.direction == 'RIGHT' else 0
        dir_down = 1 if self.direction == 'DOWN' else 0
        dir_left = 1 if self.direction == 'LEFT' else 0
        
        # Food direction
        food_up = 1 if self.food[1] < head[1] else 0
        food_right = 1 if self.food[0] > head[0] else 0
        food_down = 1 if self.food[1] > head[1] else 0
        food_left = 1 if self.food[0] < head[0] else 0
        
        # Balanced space analysis - compute for all snakes but with optimization
        next_positions = []
        current_idx = self.direction_map[self.direction]
        for action in [2, 0, 1]:  # Left, Straight, Right
            if action == 1:  # Right turn
                new_idx = (current_idx + 1) % 4
            elif action == 2:  # Left turn  
                new_idx = (current_idx - 1) % 4
            else:  # Straight
                new_idx = current_idx
            new_direction = self.directions[new_idx]
            next_pos = head.copy()
            if new_direction == 'UP':
                next_pos[1] -= 1
            elif new_direction == 'DOWN':
                next_pos[1] += 1
            elif new_direction == 'LEFT':
                next_pos[0] -= 1
            elif new_direction == 'RIGHT':
                next_pos[0] += 1
            next_positions.append(next_pos)
        
        # Calculate reachable spaces for each action
        space_left = self.get_reachable_spaces(next_positions[0]) / 50.0
        space_straight = self.get_reachable_spaces(next_positions[1]) / 50.0  
        space_right = self.get_reachable_spaces(next_positions[2]) / 50.0
        
        # Pre-compute normalizers
        width_norm = 1.0 / self.width
        height_norm = 1.0 / self.height
        grid_size_norm = 1.0 / (self.width * self.height)
        
        # Wall distances (normalized)
        wall_left = head[0] * width_norm
        wall_right = (self.width - head[0] - 1) * width_norm
        wall_up = head[1] * height_norm
        wall_down = (self.height - head[1] - 1) * height_norm
        
        # Food distance components
        food_dist_x = (self.food[0] - head[0]) * width_norm
        food_dist_y = (self.food[1] - head[1]) * height_norm
        food_dist_manhattan = (abs(self.food[0] - head[0]) + abs(self.food[1] - head[1])) / (self.width + self.height)
        
        # Snake metrics
        snake_length_norm = len(self.snake) * grid_size_norm
        
        # Tail position relative to head
        tail = self.snake[-1]
        tail_dist_x = (tail[0] - head[0]) * width_norm
        tail_dist_y = (tail[1] - head[1]) * height_norm
        
        # Fill pre-allocated array
        self.state_array[0:3] = [danger_left, danger_straight, danger_right]
        self.state_array[3:7] = [dir_left, dir_up, dir_right, dir_down]
        self.state_array[7:11] = [food_left, food_up, food_right, food_down]
        self.state_array[11:14] = [space_left, space_straight, space_right]
        self.state_array[14:18] = [wall_left, wall_right, wall_up, wall_down]
        self.state_array[18:21] = [food_dist_x, food_dist_y, food_dist_manhattan]
        self.state_array[21] = snake_length_norm
        self.state_array[22:24] = [tail_dist_x, tail_dist_y]
        
        return self.state_array.copy()
    
    def is_danger(self, action):
        # Get next head position for given action
        head = self.snake[0].copy()
        
        # Apply action to get new direction
        current_idx = self.direction_map[self.direction]
        if action == 1:  # Right turn
            new_idx = (current_idx + 1) % 4
        elif action == 2:  # Left turn
            new_idx = (current_idx - 1) % 4
        else:  # Straight
            new_idx = current_idx
        new_direction = self.directions[new_idx]
        
        # Move head in new direction
        if new_direction == 'UP':
            head[1] -= 1
        elif new_direction == 'DOWN':
            head[1] += 1
        elif new_direction == 'LEFT':
            head[0] -= 1
        elif new_direction == 'RIGHT':
            head[0] += 1
        
        # Check if new position is dangerous (use occupied set)
        if (head[0] < 0 or head[0] >= self.width or 
            head[1] < 0 or head[1] >= self.height or 
            (head[0], head[1]) in self.occupied):
            return 1
        return 0
    
    def render(self):
        if not self.render_mode:
            return
        
        self.screen.fill(self._COLOR_BLACK)
        
        # Draw snake
        for i, segment in enumerate(self.snake):
            color = self._COLOR_GREEN_HEAD if i == 0 else self._COLOR_GREEN_BODY
            rect = (segment[0] * self.cell_size, segment[1] * self.cell_size, 
                   self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, color, rect)
        
        # Draw food
        food_rect = (self.food[0] * self.cell_size, self.food[1] * self.cell_size, 
                    self.cell_size, self.cell_size)
        pygame.draw.rect(self.screen, self._COLOR_RED, food_rect)
        
        # Draw score
        score_text = self._font.render(f'Score: {self.score}', True, self._COLOR_WHITE)
        self.screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(10)  # 10 FPS
    
    def close(self):
        if self.render_mode:
            pygame.quit()