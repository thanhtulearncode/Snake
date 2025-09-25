import numpy as np
import random
from collections import deque
import pygame

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
        self.reset()
    
    def reset(self):
        # Start snake in center
        center_x, center_y = self.width // 2, self.height // 2
        self.snake = deque([[center_x, center_y]])
        # Track occupied cells for O(1) collision checks and fast food placement
        self.occupied = {(center_x, center_y)}
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
        """Optimized BFS to count spaces reachable from position"""
        # Quick rejects
        if not (0 <= pos[0] < self.width and 0 <= pos[1] < self.height):
            return 0
        if (pos[0], pos[1]) in self.occupied:
            return 0
        visited = set(self.occupied)
        queue = deque([tuple(pos)])
        visited.add(tuple(pos))
        count = 0
        # Pre-define directions to avoid repeated list creation
        directions = [(0,1), (0,-1), (1,0), (-1,0)]
        while queue and count < 50:  # Limit to prevent excessive computation
            x, y = queue.popleft()
            count += 1
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.width and 0 <= ny < self.height and 
                    (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return min(count, 50)
    
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
        new_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        reward = 0.0
        done = False
        # Food consumption - restored more nuanced rewards
        if head == self.food:
            # Base food reward scales with snake length
            base_reward = 10.0 + len(self.snake) * 0.75
            reward += base_reward
            self.score += 1
            self.steps_without_food = 0
            self.place_food()
            # Progressive milestone bonuses
            if self.score >= 10:
                reward += 8.0
            if self.score >= 20:
                reward += 15.0
            if self.score >= 30:
                reward += 25.0
            if self.score >= 40:
                reward += 40.0
        else:
            tail = self.snake.pop()
            self.occupied.discard((tail[0], tail[1]))
            # Balanced movement rewards
            if new_dist < old_dist:
                reward += 0.8  # Moving closer to food
            elif new_dist > old_dist:
                reward -= 0.4  # Moving away from food
            # Small survival reward
            reward += 0.15
            # Restored space preservation logic with balanced thresholds
            if len(self.snake) > 4:  # Only check space when snake is reasonably long
                reachable_spaces = self.get_reachable_spaces(head)
                if reachable_spaces >= 25:
                    reward += 0.6  # Good space preservation
                elif reachable_spaces >= 15:
                    reward += 0.3  # Moderate space
                elif reachable_spaces < 8:
                    reward -= 1.2  # Penalty for tight spaces
        return self.get_state(), reward, done
    
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