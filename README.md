# Implementation-of-On-Policy-Monte-Carlo-Control-using-Gymnasium
---

## Aim

To implement **Monte Carlo Control** using the Gymnasium `FrozenLake-v1` environment and learn an improved policy by estimating the action-value function from complete episodes.

---

## Problem Statement

The `FrozenLake-v1` environment consists of frozen tiles, holes, a start state, and a goal state. The agent must learn a policy that helps it reach the goal while avoiding holes.

The objective of this experiment is to:

1. Generate complete episodes using the Gymnasium environment.
2. Estimate the action-value function $Q(s,a)$ using Monte Carlo returns.
3. Use epsilon-greedy action selection for exploration and exploitation.
4. Improve the policy based on the learned Q-values.
5. Display the final Q-table, estimated state-value function, learned policy, and learning curve.

---

## Software Requirements

```bash
pip install gymnasium numpy matplotlib
```

---

## Environment Description


The **FrozenLake-v1** environment is a grid-world environment provided by Gymnasium.

The environment contains four types of tiles:

* **S** → Starting state
* **F** → Frozen surface
* **H** → Hole
* **G** → Goal

The agent starts from the starting state and must reach the goal while avoiding the holes.

For this experiment, a deterministic `4 × 4` FrozenLake environment is used with `is_slippery=False`.

The environment contains:

* **16 states**
* **4 possible actions**

The action mapping is:

| Action | Meaning |
| ------ | ------- |
| 0      | Left    |
| 1      | Down    |
| 2      | Right   |
| 3      | Up      |




## Theory

Monte Carlo methods learn from **complete episodes**. An episode is a sequence of states, actions, and rewards:

$$
S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_T
$$

The return from time step $t$ is:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
$$

Monte Carlo Control estimates the action-value function:

$$
Q(s,a)
$$

The incremental update rule is:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[G_t - Q(s,a)\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action taken in state $s$ |
| $G_t$ | Return from time step $t$ |
| $Q(s,a)$ | Action-value estimate |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |

---

## Epsilon-Greedy Policy

Monte Carlo Control uses epsilon-greedy action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1 - \epsilon$, the agent exploits by selecting the action with the highest Q-value.

The greedy action is selected as:

$$
a = \arg\max_a Q(s,a)
$$

The final learned policy is:

$$
\pi(s) = \arg\max_a Q(s,a)
$$

---

## Algorithm

1. Import the required Gymnasium, NumPy, and Matplotlib libraries.
2. Create the `FrozenLake-v1` environment.
3. Obtain the number of states and number of actions.
4. Initialize the Q-table with zeros.
5. Set the hyperparameters such as number of episodes, learning rate, discount factor, and epsilon values.
6. Use an epsilon-greedy policy to select actions.
7. Generate a complete episode until the environment reaches a terminal state.
8. Store the state, action, and reward for every step.
9. Calculate the return $G_t$ by processing the episode backwards.
10. Update the Q-value using the Monte Carlo update rule.
11. Use first-visit Monte Carlo by updating each state-action pair only once per episode.
12. Decrease epsilon after every episode while maintaining a minimum exploration value.
13. Repeat the process for all training episodes.
14. Extract the greedy policy using the maximum Q-value for every state.
15. Calculate the estimated state-value function.
16. Display the final Q-table, state-value function, learned policy, and average reward.
17. Plot the learning curve.


## Python Program

-------------------------------------------------
#### Monte Carlo Control


```python
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("FrozenLake-v1", is_slippery=False, map_name="4x4")

n_states = env.observation_space.n
n_actions = env.action_space.n

print("Number of states:", n_states)
print("Number of actions:", n_actions)

num_episodes = 20000

gamma = 0.99
alpha = 0.1

epsilon_start = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9995

max_steps_per_episode = 100

Q = np.zeros((n_states, n_actions))

episode_rewards = []

def epsilon_greedy_action(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()

    best_actions = np.flatnonzero(Q[state] == np.max(Q[state]))

    return np.random.choice(best_actions)

def generate_episode(epsilon):
    episode = []
    state, info = env.reset()
    for _ in range(max_steps_per_episode):
        action = epsilon_greedy_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if terminated or truncated:
            break
    return episode

epsilon = epsilon_start
for episode_num in range(num_episodes):
    episode = generate_episode(epsilon)
    episode_rewards.append(sum(reward for _, _, reward in episode))
    G = 0.0
    visited = set()
    for t in range(len(episode) - 1, -1, -1):
        state, action, reward = episode[t]
        G = gamma * G + reward
        if (state, action) not in visited:
            visited.add((state, action))
            Q[state, action] += alpha * (G - Q[state, action])

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if (episode_num + 1) % 2000 == 0:

        avg_reward = np.mean(episode_rewards[-1000:])

        print(
            f"Episode {episode_num + 1:5d} | "
            f"epsilon = {epsilon:.3f} | "
            f"last-1000 average reward = {avg_reward:.3f}"
        )



optimal_policy = np.argmax(Q, axis=1)

state_values = np.max(Q, axis=1)

def print_policy(policy):
    action_symbols = {0: "L", 1: "D", 2: "R", 3: "U"}
    policy_grid = np.array([action_symbols[action] for action in policy]).reshape(4, 4)
    print("Name: Sanjay Balaji S")
    print("Register Number: 212223240149")
    print("\nLearned Policy:")
    print(policy_grid)


def print_value_function(values):
    print("\nEstimated State-Value Function:")
    print(np.round(values.reshape(4, 4), 3))

print("\nFinal Q-table:")
print(np.round(Q, 3))
print_value_function(state_values)
print_policy(optimal_policy)
success_rate = np.mean(episode_rewards[-1000:])
print("\nAverage reward over last 1000 episodes:", success_rate)

window = 500
moving_average = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
plt.figure(figsize=(8, 5))
plt.plot(moving_average)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.title("Monte Carlo Control Learning Curve")
plt.grid(True)
plt.show()

env.close()

```

---

## Output


Final Q-table:

![alt text](<Screenshot 2026-09-06 210138.png>)

Estimated State-Value Function:


![alt text](<Screenshot 2026-09-06 210333.png>)




Learned Policy:



![alt text](<Screenshot 2026-09-06 210337.png>)

Average reward over last 1000 episodes: 

![alt text](<Screenshot 2026-09-06 210342.png>)


![alt text](<Screenshot 2026-09-06 210348.png>)
---

## Result
```text

The On-Policy Monte Carlo Control algorithm was successfully implemented using Gymnasium's FrozenLake-v1 environment. The agent learned an improved policy using Monte Carlo returns and an epsilon-greedy strategy.



```
---

## Inference
```text

The agent initially performs more exploration because the epsilon value starts at 1.0. As the number of training episodes increases, epsilon decreases toward the minimum value of 0.05.

The Monte Carlo algorithm calculates the return from complete episodes and uses these returns to update the Q-values.

The learned Q-table represents the estimated value of taking each action in every state. The state-value function is obtained by selecting the maximum Q-value for each state.

The epsilon-greedy policy allows the agent to explore different actions initially and gradually exploit the actions that provide higher estimated returns.

The learning curve shows the change in the average reward as the number of training episodes increases.

```







