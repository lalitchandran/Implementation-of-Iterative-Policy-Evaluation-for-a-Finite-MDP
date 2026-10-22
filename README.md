# Implementation-of-Iterative-Policy-Evaluation-for-a-Finite-MDP
## Aim

To implement iterative policy evaluation using Gymnasium and estimate the state-value function $V^\pi(s)$ for a fixed random policy.

---
## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```

---

## Environment Used

The experiment uses the **FrozenLake-v1** environment from Gymnasium.

FrozenLake is a grid-based reinforcement learning environment where the agent starts from a start state and tries to reach the goal state without falling into holes.

For the default 4 x 4 FrozenLake map:

| Component | Description |
|---|---|
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching goal, 0 otherwise |
| Terminal states | Goal and holes |

---

## Problem Statement

Evaluate a fixed random policy in the FrozenLake-v1 environment.

The agent follows a random policy, where each of the four actions is selected with equal probability:

$$
\pi(a|s) = \frac{1}{4}
$$

This probability refers to the policy's action-selection probability. The environment transition probabilities are obtained from Gymnasium using `env.P[state][action]`. If `is_slippery=True`, the agent may not move in the intended direction due to stochastic transitions.

The objective is to estimate the state-value function:

$$
V^\pi(s)
$$

---

## Theory

The state-value function under policy $pi$, denoted by $V^\pi(s)$, represents the expected return starting from state $s$ and following policy $pi$.

The Bellman expectation equation is:

```math
V^\pi(s) =
\sum_a \pi(a|s)
\sum_{s'} P(s'|s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
```

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $\pi(a \mid s)$ | Probability of selecting action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $\gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $\pi$ |

---
## Algorithm

1. Create the FrozenLake-v1 environment using Gymnasium.
2. Access the transition model of the environment.
3. Initialize \(V(s)=0\) for all states.
4. Define a random policy where each action has equal probability.
5. For each state:
   - For each action:
     - Read transition probability, next state, reward, and terminal status.
     - Apply the Bellman expectation equation.
6. Repeat until the value function converges.
7. Display the final value function as a 4 x 4 grid.

---

## Program

```python
# -------------------------------------------------
# Iterative Policy Evaluation for FrozenLake-v1
# -------------------------------------------------

import gymnasium as gym
import numpy as np

# Create the FrozenLake environment
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True)
env_model = env.unwrapped

n_states = env_model.observation_space.n
n_actions = env_model.action_space.n

print("Number of States:", n_states)
print("Number of Actions:", n_actions)
print("Actions: 0=Left, 1=Down, 2=Right, 3=Up")

# Parameters
gamma = 0.99
theta = 1e-8

# Fixed random policy: each action has equal probability
policy = np.ones((n_states, n_actions)) / n_actions


def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    """Estimate V_pi(s) using iterative policy evaluation."""
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros(n_states, dtype=float)
    iteration = 0

    while True:
        delta = 0.0
        iteration += 1

        for state in range(n_states):
            old_value = V[state]
            new_value = 0.0

            for action in range(n_actions):
                action_probability = policy[state, action]

                # Each transition is:
                # (probability, next_state, reward, terminated)
                for probability, next_state, reward, terminated in env.P[state][action]:
                    future_value = 0.0 if terminated else V[next_state]

                    new_value += action_probability * probability * (
                        reward + gamma * future_value
                    )

            V[state] = new_value
            delta = max(delta, abs(old_value - new_value))

        # Stop when the value function has converged
        if delta < theta:
            break

    return V, iteration


# Run iterative policy evaluation
V, iterations = policy_evaluation(
    env_model,
    policy,
    gamma=gamma,
    theta=theta
)

# -------------------------------------------------
# Display Output
# -------------------------------------------------

print("\\nName: __________________________")
print("Register Number: _______________")
print("Discount Factor (gamma):", gamma)
print("Convergence Threshold (theta):", theta)
print("Number of Iterations:", iterations)

print("\\nState-Value Function (16 states):")
print(np.round(V, 6))

print("\\nState-Value Function as a 4x4 Grid:")
print(np.round(V.reshape(4, 4), 4))

# Display each map tile with its value
print("\\nGrid with tile labels and state values:")
print("S = Start, F = Frozen, H = Hole, G = Goal")
desc = env_model.desc.astype("U1")

for row in range(4):
    row_text = []
    for col in range(4):
        state = row * 4 + col
        row_text.append(f"{desc[row, col]}:{V[state]:.4f}")
    print("  ".join(row_text))

env.close()
```
---

## Output

```text
<img width="772" height="385" alt="image" src="https://github.com/user-attachments/assets/fe7c223e-bbbc-47f1-9b0d-20cdad998f58" />

```
---

## Result

Iterative policy evaluation was implemented successfully using the Gymnasium FrozenLake environment. The state-value function for the fixed random policy was estimated using the Bellman expectation equation.

---

## Inference

```text

Iterative policy evaluation estimates how good each state is when the agent follows the specified fixed policy.

States with more favorable expected routes to the goal generally have higher values.

States near holes or with less favorable routes generally have lower values.

Because is_slippery=True, the environment has stochastic transitions, which are included in the calculation through env.P[state][action].

The random policy selects every action with equal probability, so it does not intentionally prefer the safest or shortest route.

The estimated values depend on the policy, transition model, discount factor, and convergence threshold.

These values describe the selected random policy; they are not the optimal state values.

```




---


