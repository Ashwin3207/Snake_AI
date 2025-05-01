import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any
from dqn_agent import DQNAgent
from snake_env import SnakeEnv


def train(
    n_games: int = 1000,
    model_path: str = "models/best_model.pth",
    save_interval: int = 100,
    target_score: int = 50,
    render_interval: int = 100,
    warmup_games: int = 50
) -> Dict[str, Any]:
    """Train the DQN agent."""
    Path("models").mkdir(exist_ok=True)
    env = SnakeEnv(render_mode=None)
    agent = DQNAgent()
    
    if Path(model_path).exists():
        print(f"Loading existing model from {model_path}")
        agent.load_model(model_path)
    
    stats = {
        'scores': [],
        'mean_scores': [],
        'best_score': 0,
        'epsilon_history': [],
        'loss_history': [],
        'start_time': time.time()
    }
    
    print(f"Training for {n_games} games...")
    for game in range(1, n_games + 1):
        render = game % render_interval == 0
        if render:
            env = SnakeEnv(render_mode="human")
        
        state = env.reset()
        done = False
        score = 0
        episode_loss = []
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            
            if game > warmup_games:
                loss = agent.train_short_memory(state, action, reward, next_state, done)
                if loss:
                    episode_loss.append(loss)
            
            state = next_state
            score = info['score']
            
            if render:
                env.render()
                time.sleep(0.02)
        
        if game > warmup_games:
            agent.train_long_memory()
            agent.decay_epsilon()
        
        if render:
            env.close()
            env = SnakeEnv(render_mode=None)
        
        stats['scores'].append(score)
        stats['epsilon_history'].append(agent.epsilon)
        mean_score = np.mean(stats['scores'][-100:])
        stats['mean_scores'].append(mean_score)
        
        if episode_loss:
            stats['loss_history'].append(np.mean(episode_loss))
        
        if score > stats['best_score']:
            stats['best_score'] = score
            agent.save_model(model_path)
            if score >= target_score:
                print(f"\nTarget score of {target_score} reached!")
                break
        
        if game % save_interval == 0:
            agent.save_model(f"models/model_game_{game}.pth")
        
        print(f"Game {game:4d} | Score: {score:3d} | Best: {stats['best_score']:3d} | "
              f"Mean: {mean_score:5.1f} | Epsilon: {agent.epsilon:.3f}")

    _plot_training(stats)
    print(f"\nTraining completed in {(time.time()-stats['start_time'])/60:.1f} minutes")
    return stats


def _plot_training(stats: Dict[str, Any]) -> None:
    """Plot training statistics."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(stats['scores'], alpha=0.6, label='Scores')
    plt.plot(stats['mean_scores'], label='Mean (100 games)')
    plt.title("Training Scores")
    plt.xlabel("Games")
    plt.ylabel("Score")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(stats['epsilon_history'], label='Epsilon')
    if stats['loss_history']:
        plt.plot(stats['loss_history'], label='Loss')
    plt.title("Training Metrics")
    plt.xlabel("Games")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_stats.png")
    plt.show()


if __name__ == "__main__":
    train()