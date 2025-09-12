import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
from datetime import datetime
import pygame
from game_env import SnakeEnv
from model import Agent
import glob
import os
import json

def get_latest_model(models_dir="models", pattern="snake_model_*.pth"):
    search_path = os.path.join(models_dir, pattern)
    files = glob.glob(search_path)
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    return latest_file

def train(episodes=1000, render=False, model_dir="models", plot_dir="plots"):
    """Train the DQN agent."""
    env = SnakeEnv(render=render)
    agent = Agent()
    scores = deque(maxlen=100)
    avg_scores = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Training for {episodes} episodes...")
    for episode in range(episodes):
        state = env.reset()
        total_score = 0
        steps = 0
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_score += reward
            steps += 1
            if render:
                env.render()
                time.sleep(0.1)
            if done:
                break
        scores.append(env.score)
        # Train the agent
        if len(agent.memory) > 64:
            agent.replay(64)
        # Update target network every 100 episodes
        if episode % 100 == 0:
            agent.update_target_network()
        # Logging
        if episode % 100 == 0:
            avg_score = np.mean(scores)
            avg_scores.append(avg_score)
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, "
                  f"Epsilon: {agent.epsilon:.3f}, Steps: {steps}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    # Save the trained model
    model_filename = os.path.join(model_dir, f"snake_model_{timestamp}.pth")
    agent.save(model_filename)
    print(f"Model saved as '{model_filename}'")
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, len(avg_scores) * 100, 100), avg_scores)
    plt.title('Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Average Score (last 100 episodes)')
    plt.grid(True)
    plot_filename = os.path.join(plot_dir, f"training_progress_{timestamp}.png")
    plt.savefig(plot_filename)
    plt.show()
    print(f"Training curve saved as '{plot_filename}'")
    env.close()
    return agent

def play(model_path=None, num_games=5):
    """Play the game with trained agent."""
    env = SnakeEnv(render=True)
    agent = Agent()
    if model_path is None:
        model_path = get_latest_model()
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Using untrained agent.")
    scores = []
    for game in range(num_games):
        state = env.reset()
        total_score = 0
        steps = 0
        print(f"Game {game + 1}/{num_games} started")
        while True:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        env.close()
                        return
            action = agent.act(state, training=False)  # No exploration
            state, reward, done = env.step(action)
            total_score += reward
            steps += 1
            env.render()
            if done:
                break
        final_score = env.score
        scores.append(final_score)
        print(f"Game {game + 1} finished - Score: {final_score}, Steps: {steps}")
        # Wait 2 seconds before next game
        time.sleep(2)
    if scores:
        print(f"Average score: {np.mean(scores):.2f}")
        print(f"Best score: {max(scores)}")
    env.close()

def evaluate(model_path=None, episodes=100, save_results=True, results_dir="results"):
    """Evaluate the trained agent."""
    env = SnakeEnv(render=False)
    agent = Agent()
    if model_path is None:
        model_path = get_latest_model()
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model file {model_path} not found. Using untrained agent.")
    scores = []
    print(f"Evaluating for {episodes} episodes...")
    for episode in range(episodes):
        state = env.reset()
        while True:
            action = agent.act(state, training=False)
            state, reward, done = env.step(action)
            if done:
                break
        scores.append(env.score)
        if (episode + 1) % 20 == 0:
            avg_score = np.mean(scores[-20:])
            print(f"Episodes {episode - 19}-{episode}: Average Score: {avg_score:.2f}")
    avg_score = float(np.mean(scores))
    best_score = int(max(scores))
    success_rate = float(sum(1 for s in scores if s > 0) / len(scores) * 100)
    print(f"\nEvaluation Results:")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Best Score: {best_score}")
    print(f"Success Rate (score > 0): {success_rate:.1f}%")
    # Save results
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(results_dir, f"eval_{timestamp}.json")
        results = {
            "model": model_path,
            "episodes": episodes,
            "average_score": avg_score,
            "best_score": best_score,
            "success_rate": success_rate,
            "scores": scores,
            "timestamp": timestamp
        }
        with open(result_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {result_file}")
    return avg_score, best_score, success_rate

def main():
    parser = argparse.ArgumentParser(description='Simple Snake RL')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'play', 'eval'],
                        help='Mode: train, play, or eval')
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of episodes for training/evaluation')
    parser.add_argument('--model', type=str, default=None,
                        help='Model file path (default: latest)')
    parser.add_argument('--render', action='store_true',
                        help='Render during training (slower)')
    parser.add_argument('--games', type=int, default=5,
                        help='Number of games to play in play mode')
    args = parser.parse_args()
    if args.mode == 'train':
        train(episodes=args.episodes, render=args.render)
    elif args.mode == 'play':
        play(model_path=args.model, num_games=args.games)
    elif args.mode == 'eval':
        evaluate(model_path=args.model, episodes=args.episodes)

if __name__ == "__main__":
    main()