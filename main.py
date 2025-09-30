import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import pygame
from game_env import SnakeEnv
from model import Agent
import glob
import os
import json
from datetime import datetime

def get_latest_model(models_dir="models", pattern="snake_model_*.pth"):
    """Find the most recently created model file."""
    search_path = os.path.join(models_dir, pattern)
    files = glob.glob(search_path)
    if not files:
        return None
    return max(files, key=os.path.getctime)

def train(episodes=5000, render=False, model_dir="models", plot_dir="plots"):
    """Train the DQN agent with optimized settings."""
    print(f"Starting training for {episodes} episodes...")
    
    # Initialize environment and agent
    env = SnakeEnv(render=render)
    agent = Agent()
    
    # Training tracking
    scores = deque(maxlen=100)
    avg_scores = []
    best_avg_score = 0
    training_start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Device: {agent.device}")
    print(f"Network parameters: {sum(p.numel() for p in agent.q_network.parameters())}")
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        # Episode loop
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
                time.sleep(0.01)
            
            if done:
                break
        
        scores.append(env.score)
        
        # Train the agent multiple times per episode for faster learning
        if len(agent.replay_buffer) > 2000:
            if episode % 2 == 0:
                for _ in range(4):
                    agent.replay(batch_size=256)
        
        # Progress logging every 50 episodes
        if episode % 50 == 0 and episode > 0:
            avg_score = np.mean(scores)
            total_time = time.time() - training_start_time
            
            print(f"Episode {episode:4d} | "
                  f"Avg Score: {avg_score:5.2f} | "
                  f"Best: {max(scores):2d} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Time: {total_time/60:.1f}min")
            
            avg_scores.append(avg_score)
            
            # Save best model
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                os.makedirs(model_dir, exist_ok=True)
                best_model_path = os.path.join(model_dir, f"best_snake_{timestamp}.pth")
                agent.save(best_model_path)
                print(f"  → New best average: {avg_score:.2f}")
    
    # Save final model and create plots
    _save_final_results(agent, scores, avg_scores, timestamp, model_dir, plot_dir, training_start_time)
    env.close()
    return agent

def _save_final_results(agent, scores, avg_scores, timestamp, model_dir, plot_dir, start_time):
    """Save final model and generate training plots."""
    # Create directories
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save final model
    model_filename = os.path.join(model_dir, f"snake_model_{timestamp}.pth")
    agent.save(model_filename)
    print(f"Final model saved: {model_filename}")
    
    # Generate comprehensive training plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Average scores over time
    if avg_scores:
        axes[0, 0].plot(range(0, len(avg_scores) * 50, 50), avg_scores, 'b-', linewidth=2)
        axes[0, 0].set_title('Average Score Progress')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Average Score (last 100)')
        axes[0, 0].grid(True, alpha=0.3)
    
    # Recent individual scores
    recent_scores = list(scores)[-min(1000, len(scores)):]
    axes[0, 1].plot(recent_scores, alpha=0.7, color='green')
    axes[0, 1].set_title('Recent Individual Scores')
    axes[0, 1].set_xlabel('Recent Episodes')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Score distribution histogram
    axes[1, 0].hist(scores, bins=min(20, len(set(scores))), alpha=0.7, color='orange')
    axes[1, 0].set_title('Score Distribution')
    axes[1, 0].set_xlabel('Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training statistics
    total_time = time.time() - start_time
    stats_text = f"""Training Summary:
    
Episodes: {len(scores)}
Final Avg Score: {np.mean(scores):.2f}
Best Single Score: {max(scores)}
Success Rate: {sum(1 for s in scores if s > 0) / len(scores) * 100:.1f}%
Final Epsilon: {agent.epsilon:.3f}
Training Time: {total_time/60:.1f} minutes
Episodes/min: {len(scores)/(total_time/60):.1f}
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                    verticalalignment='center', fontfamily='monospace', fontsize=10)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plot_filename = os.path.join(plot_dir, f"training_results_{timestamp}.png")
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Training plots saved: {plot_filename}")

def play(model_path=None, num_games=5):
    """Play games with the trained agent."""
    print(f"Starting {num_games} games with trained agent...")
    
    env = SnakeEnv(render=True)
    agent = Agent()
    
    # Load model
    if model_path is None:
        model_path = get_latest_model()
    
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model: {model_path}")
    else:
        print("No trained model found. Using random agent.")
    
    scores = []
    
    for game in range(num_games):
        print(f"\nStarting Game {game + 1}/{num_games}")
        
        state = env.reset()
        steps = 0
        
        while True:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    env.close()
                    return
            
            action = agent.act(state, training=False)
            state, reward, done = env.step(action)
            steps += 1
            env.render()
            
            if done:
                break
        
        score = env.score
        scores.append(score)
        print(f"Game {game + 1} finished - Score: {score}, Steps: {steps}")
        
        if game < num_games - 1:  # Don't wait after last game
            time.sleep(1)
    
    # Print summary
    if scores:
        print(f"\nGame Summary:")
        print(f"Average Score: {np.mean(scores):.2f}")
        print(f"Best Score: {max(scores)}")
        print(f"All Scores: {scores}")
    
    env.close()

def evaluate(model_path=None, episodes=100, save_results=True, results_dir="results"):
    """Evaluate the trained agent over multiple episodes."""
    print(f"Evaluating agent for {episodes} episodes...")
    
    env = SnakeEnv(render=False)
    agent = Agent()
    
    # Load model
    if model_path is None:
        model_path = get_latest_model()
    
    if model_path and os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model: {model_path}")
    else:
        print("No trained model found. Using random agent.")
    
    scores = []
    start_time = time.time()
    
    # Run evaluation episodes
    for episode in range(episodes):
        state = env.reset()
        
        while True:
            action = agent.act(state, training=False)
            state, reward, done = env.step(action)
            if done:
                break
        
        scores.append(env.score)
        
        # Progress update every 25 episodes
        if (episode + 1) % 25 == 0:
            recent_avg = np.mean(scores[-25:])
            elapsed = time.time() - start_time
            speed = (episode + 1) / elapsed
            print(f"Episodes {episode-24:3d}-{episode:3d}: "
                  f"Avg Score: {recent_avg:.2f} | "
                  f"Speed: {speed:.1f} eps/sec")
    
    # Calculate final statistics
    eval_time = time.time() - start_time
    avg_score = np.mean(scores)
    best_score = max(scores)
    success_rate = sum(1 for s in scores if s > 0) / len(scores) * 100
    
    print(f"\nEvaluation Results:")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Best Score: {best_score}")
    print(f"Success Rate (>0): {success_rate:.1f}%")
    print(f"Evaluation Speed: {episodes/eval_time:.1f} episodes/sec")
    
    # Save results
    if save_results:
        _save_evaluation_results(model_path, episodes, scores, avg_score, 
                                best_score, success_rate, eval_time, results_dir)
    
    env.close()
    return avg_score, best_score, success_rate

def _save_evaluation_results(model_path, episodes, scores, avg_score, 
                           best_score, success_rate, eval_time, results_dir):
    """Save evaluation results to JSON file."""
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        "model_path": model_path,
        "episodes": episodes,
        "average_score": float(avg_score),
        "best_score": int(best_score),
        "success_rate": float(success_rate),
        "evaluation_time": float(eval_time),
        "episodes_per_second": float(episodes / eval_time),
        "all_scores": scores,
        "timestamp": timestamp
    }
    
    result_file = os.path.join(results_dir, f"evaluation_{timestamp}.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved: {result_file}")

def benchmark():
    """Benchmark training speed."""
    print("Benchmarking training speed...")
    
    env = SnakeEnv(render=False)
    agent = Agent()
    
    # Warm-up phase
    print("Warming up...")
    for _ in range(10):
        state = env.reset()
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        if len(agent.memory) > 100:
            agent.replay(32)
    
    # Actual benchmark
    print("Running benchmark...")
    start_time = time.time()
    num_episodes = 100
    total_steps = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_steps = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_steps += 1
            
            if done:
                break
        
        total_steps += episode_steps
        
        # Train periodically
        if len(agent.memory) > 500 and episode % 5 == 0:
            agent.replay(32)
    
    benchmark_time = time.time() - start_time
    
    print(f"\nBenchmark Results:")
    print(f"Episodes: {num_episodes}")
    print(f"Total Steps: {total_steps}")
    print(f"Total Time: {benchmark_time:.2f}s")
    print(f"Episodes/sec: {num_episodes/benchmark_time:.2f}")
    print(f"Steps/sec: {total_steps/benchmark_time:.1f}")
    print(f"Avg steps/episode: {total_steps/num_episodes:.1f}")
    
    env.close()

def main():
    parser = argparse.ArgumentParser(description='Optimized Snake RL Training')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'play', 'eval', 'benchmark'],
                       help='Mode: train, play, eval, or benchmark')
    parser.add_argument('--episodes', type=int, default=5000,
                       help='Number of episodes for training/evaluation')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to model file (default: use latest)')
    parser.add_argument('--render', action='store_true',
                       help='Show game rendering during training')
    parser.add_argument('--games', type=int, default=5,
                       help='Number of games to play in play mode')
    
    args = parser.parse_args()
    
    print(f"Snake RL - Mode: {args.mode}")
    print("-" * 50)
    
    if args.mode == 'train':
        train(episodes=args.episodes, render=args.render)
    elif args.mode == 'play':
        play(model_path=args.model, num_games=args.games)
    elif args.mode == 'eval':
        evaluate(model_path=args.model, episodes=args.episodes)
    elif args.mode == 'benchmark':
        benchmark()
    
    print("Done!")

if __name__ == "__main__":
    main()