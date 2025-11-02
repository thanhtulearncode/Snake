# Snake RL

A reinforcement learning implementation of the classic Snake game using Deep Q-Network (DQN). The agent learns to play Snake through self-play using PyTorch.

## Features

- **Dueling DQN Architecture**: Advanced neural network with value and advantage streams
- **Prioritized Experience Replay**: Focuses learning on important experiences
- **N-step Returns**: Better value estimation through multi-step returns
- **Optimized Environment**: Fast game engine with collision detection and visual rendering
- **Multiple Training Modes**: Train, play, evaluate, and benchmark the agent

## Performance

Average score: ~28 | Best: 41+ | Success rate: 100%

## Installation

1. Clone and enter directory:

```bash
git clone <https://github.com/thanhtulearncode/Snake>
cd Snake
```

2. Create virtual environment:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train agent
python main.py --mode train --episodes 5000

# Watch agent play
python main.py --mode play --games 5

# Evaluate performance
python main.py --mode eval --episodes 500

# Benchmark speed
python main.py --mode benchmark
```

**Arguments:**

- `--mode`: train, play, eval, or benchmark
- `--episodes`: number of episodes (default: 5000)
- `--render`: show game during training
- `--games`: number of games in play mode (default: 5)

## Technical Details

**Algorithm:** Dueling DQN with n-step returns, prioritized replay, and double Q-learning

**State:** 24 features (danger detection, direction, food position, space analysis)

**Actions:** 3 (turn left, straight, turn right)

**Rewards:** Food consumption (+10 + length bonus), approaching food (+0.8), collisions (-10)
