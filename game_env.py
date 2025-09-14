import numpy as np
import random
from collections import deque
import heapq
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
        self.reset()
    
    def reset(self):
        # Start snake in center
        center_x, center_y = self.width // 2, self.height // 2
        self.snake = deque([[center_x, center_y]])
        self.direction = 'RIGHT'
        self.place_food()
        self.score = 0
        self.steps_without_food = 0
        return self.get_state()
    
    def place_food(self):
        while True:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            if [x, y] not in self.snake:
                self.food = [x, y]
                break

    
    def get_reachable_spaces(self, pos):
        """BFS to count spaces reachable from position"""
        visited = set(tuple(segment) for segment in self.snake)
        if tuple(pos) in visited:
            return 0
        queue = deque([tuple(pos)])
        visited.add(tuple(pos))
        count = 0
        while queue and count < 50:  # Limit to prevent excessive computation
            x, y = queue.popleft()
            count += 1
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
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
        # Timeout penalty
        if self.steps_without_food > self.width * self.height * 4:
            return self.get_state(), -5.0, True
        self.update_direction(action)
        self.move_snake()
        head = self.snake[0]
        new_dist = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        reward = 0.0
        done = False
        # Collision penalties
        if (head[0] < 0 or head[0] >= self.width or 
            head[1] < 0 or head[1] >= self.height):
            reward = -10.0
            done = True
            return self.get_state(), reward, done
        if head in list(self.snake)[1:]:
            reward = -10.0
            done = True
            return self.get_state(), reward, done
        # Food consumption
        if head == self.food:
            # Base food reward scales with snake length
            base_reward = 10.0 + len(self.snake) * 0.5
            reward += base_reward
            self.score += 1
            self.steps_without_food = 0
            self.place_food()
            # Bonus for achieving milestones
            if self.score >= 10:
                reward += 5.0
            if self.score >= 20:
                reward += 10.0
            if self.score >= 30:
                reward += 20.0
        else:
            self.snake.pop()
            # Distance-based reward (improved)
            if new_dist < old_dist:
                reward += 1.0  # Moving closer to food
            elif new_dist > old_dist:
                reward -= 0.5  # Moving away from food
            # Small survival reward
            reward += 0.1
            # Space preservation reward
            reachable_spaces = self.get_reachable_spaces(head)
            if reachable_spaces >= 20:
                reward += 0.5
            elif reachable_spaces < 5:
                reward -= 1.0  # Penalty for getting into tight spaces
        # Perfect game bonus
        if len(self.snake) == self.width * self.height:
            reward += 500.0
            done = True
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
        # Basic danger detection (keep from original)
        danger_straight = self.is_danger(0)
        danger_right = self.is_danger(1)
        danger_left = self.is_danger(2)
        # Direction encoding (keep from original)
        dir_up = 1 if self.direction == 'UP' else 0
        dir_right = 1 if self.direction == 'RIGHT' else 0
        dir_down = 1 if self.direction == 'DOWN' else 0
        dir_left = 1 if self.direction == 'LEFT' else 0
        # Food direction (keep from original)
        food_up = 1 if self.food[1] < head[1] else 0
        food_right = 1 if self.food[0] > head[0] else 0
        food_down = 1 if self.food[1] > head[1] else 0
        food_left = 1 if self.food[0] < head[0] else 0
        # Enhanced space analysis - key improvement
        next_positions = []
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        current_idx = directions.index(self.direction)
        for action in [2, 0, 1]:  # Left, Straight, Right
            if action == 1:  # Right turn
                new_idx = (current_idx + 1) % 4
            elif action == 2:  # Left turn  
                new_idx = (current_idx - 1) % 4
            else:  # Straight
                new_idx = current_idx
            new_direction = directions[new_idx]
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
        # Wall distances (normalized)
        wall_left = head[0] / self.width
        wall_right = (self.width - head[0] - 1) / self.width
        wall_up = head[1] / self.height
        wall_down = (self.height - head[1] - 1) / self.height
        # Food distance components
        food_dist_x = (self.food[0] - head[0]) / self.width
        food_dist_y = (self.food[1] - head[1]) / self.height
        food_dist_manhattan = (abs(self.food[0] - head[0]) + abs(self.food[1] - head[1])) / (self.width + self.height)
        # Snake metrics
        snake_length_norm = len(self.snake) / (self.width * self.height)
        # Tail position relative to head
        tail = self.snake[-1]
        tail_dist_x = (tail[0] - head[0]) / self.width
        tail_dist_y = (tail[1] - head[1]) / self.height
        state = np.array([
            # Danger signals (3)
            danger_left, danger_straight, danger_right,
            # Current direction (4) 
            dir_left, dir_up, dir_right, dir_down,
            # Food direction (4)
            food_left, food_up, food_right, food_down,
            # Space analysis - KEY FEATURE (3)
            space_left, space_straight, space_right,
            # Wall distances (4)
            wall_left, wall_right, wall_up, wall_down,
            # Food distance metrics (3)
            food_dist_x, food_dist_y, food_dist_manhattan,
            # Snake metrics (1)
            snake_length_norm,
            # Tail tracking (2)
            tail_dist_x, tail_dist_y
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