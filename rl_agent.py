# rl_agent.py
"""
Módulo RL simple: GridWorld + Q-Learning.
Funciones principales:
 - train_rl(params): entrena y devuelve resultados + imágenes en base64.
 - simulate_model(): simula 1 episodio usando el modelo guardado.
Guarda el Q-table en models/q_table.pkl
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle, os, io, base64, json, time

os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ---------------- GridWorld ----------------
class GridWorld:
    def __init__(self, size=5, start=(0,0), goal=None, obstacles=None):
        self.size = size
        self.start = start
        self.agent_pos = start
        self.goal = goal if goal else (size-1, size-1)
        self.obstacles = obstacles if obstacles else []
        self.action_space = [0,1,2,3]  # up, right, down, left
        self.n_states = size * size

    def reset(self):
        self.agent_pos = self.start
        return self._pos_to_state(self.agent_pos)

    def step(self, action):
        x,y = self.agent_pos
        if action == 0: x = max(0, x-1)
        elif action == 1: y = min(self.size-1, y+1)
        elif action == 2: x = min(self.size-1, x+1)
        elif action == 3: y = max(0, y-1)
        new_pos = (x,y)

        # obstacle handling: penaliza y no se mueve
        if new_pos in self.obstacles:
            reward = -5.0
            done = False
            new_pos = self.agent_pos  # stays in place
        elif new_pos == self.goal:
            reward = 10.0
            done = True
        else:
            reward = -0.1
            done = False

        self.agent_pos = new_pos
        return self._pos_to_state(new_pos), reward, done, {}

    def _pos_to_state(self, pos):
        x,y = pos
        return x * self.size + y

    def state_to_pos(self, s):
        x = s // self.size
        y = s % self.size
        return (x,y)

# ---------------- Q-Learning ----------------
def q_learning_train(env, episodes=1000, alpha=0.1, gamma=0.99,
                     epsilon=0.3, epsilon_decay=0.995, max_steps=200):
    n_states = env.n_states
    n_actions = len(env.action_space)
    Q = np.zeros((n_states, n_actions), dtype=float)
    rewards_per_episode = []

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        for step in range(max_steps):
            if np.random.rand() < epsilon:
                action = np.random.choice(n_actions)
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, done, _ = env.step(action)
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

            state = next_state
            total_reward += reward
            if done:
                break

        rewards_per_episode.append(total_reward)
        epsilon *= epsilon_decay
    return Q, rewards_per_episode

# ---------------- plotting helpers ----------------
def _img_from_plt(plt_fig):
    buf = io.BytesIO()
    plt_fig.savefig(buf, format='png', bbox_inches='tight')
    plt_fig.clf()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

def plot_rewards_b64(rewards, window=50):
    fig = plt.figure(figsize=(6,3))
    plt.title("Recompensa por episodio (promedio móvil)")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    if len(rewards) >= window:
        avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(len(avg)), avg)
    else:
        plt.plot(rewards)
    plt.tight_layout()
    b64 = _img_from_plt(fig)
    plt.close(fig)
    return b64

def plot_trajectory_b64(env, Q, max_steps=200):
    # simula un episodio greedy y dibuja la trayectoria
    sim_env = GridWorld(size=env.size, start=env.start, goal=env.goal, obstacles=env.obstacles)
    state = sim_env.reset()
    traj = [sim_env.state_to_pos(state)]
    for _ in range(max_steps):
        action = int(np.argmax(Q[state]))
        next_state, reward, done, _ = sim_env.step(action)
        traj.append(sim_env.state_to_pos(next_state))
        state = next_state
        if done: break

    fig, ax = plt.subplots(figsize=(4,4))
    size = sim_env.size
    ax.set_xlim(-0.5, size-0.5)
    ax.set_ylim(-0.5, size-0.5)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.invert_yaxis()
    ax.grid(True)

    # dibujar obstáculos
    for (ox,oy) in sim_env.obstacles:
        rect = plt.Rectangle((oy-0.5, ox-0.5), 1,1, color='black')
        ax.add_patch(rect)

    xs = [p[1] for p in traj]
    ys = [p[0] for p in traj]
    ax.plot(xs, ys, marker='o')
    sx, sy = sim_env.start
    gx, gy = sim_env.goal
    ax.text(sy, sx, 'Start', va='center', ha='center', bbox=dict(facecolor='white', alpha=0.7))
    ax.text(gy, gx, 'Goal', va='center', ha='center', bbox=dict(facecolor='white', alpha=0.7))
    plt.tight_layout()
    b64 = _img_from_plt(fig)
    plt.close(fig)
    return b64

# ---------------- public API ----------------
def train_rl(params):
    """
    params: dict con keys opcionales:
      - episodes (int)
      - alpha (float)
      - gamma (float)
      - epsilon (float)
      - epsilon_decay (float)
      - size (int)
      - obstacles (list of (x,y))
      - start (tuple)
      - goal (tuple)
    Devuelve dict con resultados y dos imágenes en base64 (reward_plot, trajectory_plot).
    Guarda modelo en models/q_table.pkl
    """
    episodes = int(params.get('episodes', 1000))
    alpha = float(params.get('alpha', 0.1))
    gamma = float(params.get('gamma', 0.99))
    epsilon = float(params.get('epsilon', 0.3))
    epsilon_decay = float(params.get('epsilon_decay', 0.995))
    size = int(params.get('size', 5))
    obstacles = params.get('obstacles', [(1,1), (2,2), (3,1)])  # ejemplo por defecto
    start = tuple(params.get('start', (0,0)))
    goal = tuple(params.get('goal', (size-1, size-1)))

    env = GridWorld(size=size, start=start, goal=goal, obstacles=obstacles)

    start_time = time.time()
    Q, rewards = q_learning_train(env, episodes=episodes, alpha=alpha,
                                 gamma=gamma, epsilon=epsilon,
                                 epsilon_decay=epsilon_decay)
    elapsed = time.time() - start_time

    # guardar modelo
    model_path = 'models/q_table.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'Q': Q,
            'env_size': size,
            'obstacles': obstacles,
            'start': start,
            'goal': goal
        }, f)

    # gráficas b64
    reward_b64 = plot_rewards_b64(rewards, window=max(1, episodes//20))
    traj_b64 = plot_trajectory_b64(env, Q)

    results = {
        'episodes': episodes,
        'alpha': alpha,
        'gamma': gamma,
        'epsilon_initial': epsilon,
        'epsilon_decay': epsilon_decay,
        'elapsed_seconds': elapsed,
        'avg_reward_last_100': float(np.mean(rewards[-100:])) if len(rewards) >= 1 else None,
        'model_path': model_path
    }

    # guardar resumen
    with open('results/train_info.json', 'w') as f:
        json.dump(results, f, indent=2)

    return {
        'status': 'ok',
        'results': results,
        'reward_plot': reward_b64,
        'trajectory_plot': traj_b64,
        'raw_rewards': rewards  # opcional, útil para debugging
    }

def simulate_model():
    """
    Simula un episodio usando el modelo guardado en models/q_table.pkl
    Devuelve total_reward y trajectory_plot (b64).
    """
    model_path = 'models/q_table.pkl'
    if not os.path.exists(model_path):
        return {'status': 'error', 'message': 'No model found. Train first.'}

    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    Q = data['Q']
    size = data['env_size']
    obstacles = data['obstacles']
    start = tuple(data['start'])
    goal = tuple(data['goal'])
    env = GridWorld(size=size, start=start, goal=goal, obstacles=obstacles)

    state = env.reset()
    total_reward = 0.0
    for _ in range(200):
        action = int(np.argmax(Q[state]))
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        if done:
            break

    traj_b64 = plot_trajectory_b64(env, Q)
    return {'status':'ok', 'total_reward': total_reward, 'trajectory_plot': traj_b64}
