# [Paper Materials] From Two-Dimensional to Three-Dimensional Environment with Q-Learning: Modeling Autonomous Navigation with Reinforcement Learning and no Libraries

### Dataset generated in *"From Two-Dimensional to Three-Dimensional Environment with Q-Learning: Modeling Autonomous Navigation with Reinforcement Learning and no Libraries"*, available at [arXiv:2402.05867](https://arxiv.org/abs/2402.05867) and at [math.paperswithcode](https://math.paperswithcode.com/paper/exploring-pseudorandom-value-addition).

**Abstract:** Reinforcement learning (RL) algorithms have become indispensable tools in artificial intelligence, empowering agents to acquire optimal decision-making policies through interactions with their environment and feedback mechanisms. This study explores the performance of RL agents in both two-dimensional (2D) and three-dimensional (3D) environments, aiming to research the dynamics of learning across different spatial dimensions. A key aspect of this investigation is the absence of pre-made libraries for learning, with the algorithm developed exclusively through computational mathematics. The methodological framework centers on RL principles, employing a Q-learning agent class and distinct environment classes tailored to each spatial dimension. The research aims to address the question: How do reinforcement learning agents adapt and perform in environments of varying spatial dimensions, particularly in 2D and 3D settings? Through empirical analysis, the study evaluates agents' learning trajectories and adaptation processes, revealing insights into the efficacy of RL algorithms in navigating complex, multi-dimensional spaces. Reflections on the findings prompt considerations for future research, particularly in understanding the dynamics of learning in higher-dimensional environments.

## About the code behind

This code defines a Q-learning agent class used for reinforcement learning tasks. Upon initialization, the agent is configured with parameters such as the number of possible actions, learning rate, discount factor, and exploration rate (epsilon). The Q-learning agent maintains a Q-table to store Q-values for state-action pairs. The get_q_value method retrieves the Q-value for a given state-action pair. The choose_action method implements an epsilon-greedy policy to select actions, balancing exploration and exploitation. The update_q_value method updates Q-values based on the observed reward and the transition to the next state, following the Q-learning update rule. Overall, this class encapsulates the essential functionalities required for a Q-learning agent to interact with an environment, learn from experiences, and improve its policy over time.

```python
class QLearningAgent:
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}


    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)


    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            best_action = None
            best_q_value = float('-inf')
            for action in range(self.num_actions):
                q_value = self.get_q_value(state, action)
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
            return best_action


    def update_q_value(self, state, action, reward, next_state):
        best_next_action = max([self.get_q_value(next_state, next_action) for next_action in range(self.num_actions)])
        td_target = reward + self.discount_factor * best_next_action
        td_delta = td_target - self.get_q_value(state, action)
        self.q_table[(state, action)] = self.get_q_value(state, action) + self.learning_rate * td_delta
```

This function train_agent is responsible for training a reinforcement learning agent within a given environment over a specified number of episodes, with a limit on the maximum number of steps per episode. During training, it tracks the rewards and the number of steps taken per episode. It iterates over each episode, initializing the state to the environment's starting point and then enters a loop to interact with the environment until the episode ends. Within each episode, the agent selects an action based on its policy, executes the action in the environment, observes the resulting state and reward, and updates its Q-values accordingly using the Q-learning update rule. The function terminates the episode if the agent reaches the goal state or if the maximum number of steps is reached. After each episode, it records the total reward obtained and the number of steps taken. Finally, it returns the lists of rewards and steps per episode, providing insights into the agent's learning progress over the process.

```python
def train_agent(agent, env, num_episodes, max_steps):
    rewards_per_episode = []
    steps_per_episode = []
    for episode in range(num_episodes):
        state = env.start
        total_reward = 0
        num_steps = 0
        while True:
            action = agent.choose_action(state)
            next_state = env.take_action(state, action)
            reward = env.get_reward(next_state)
            agent.update_q_value(state, action, reward, next_state)
            total_reward += reward
            num_steps += 1
            state = next_state
            if next_state == env.goal or num_steps == max_steps:
                rewards_per_episode.append(total_reward)
                steps_per_episode.append(num_steps)
                print(f"Episode {episode + 1}: Total Reward = {total_reward}, Total Steps = {num_steps}")
                break
    return rewards_per_episode, steps_per_episode
```

**2D:** This code defines an Environment class representing a 2D environment for a reinforcement learning task. It includes methods to initialize the environment with its dimensions, starting point, and goal location. The is_valid_position method checks whether a given position is within the boundaries of the environment. The get_reward method returns a reward of 1.0 if the agent reaches the goal state; otherwise, it returns a reward of 0.0. The take_action method updates the agent's position based on the chosen action, considering movement in four directions: up, down, left, and right, ensuring the new position remains within the boundaries.

```python
class Environment:
    def __init__(self, width, height, start, goal):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal


    def is_valid_position(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height


    def get_reward(self, state):
        if state == self.goal:
            return 1.0
        else:
            return 0.0


    def take_action(self, state, action):
        x, y = state
        if action == 0:  # up
            y += 1
        elif action == 1:  # down
            y -= 1
        elif action == 2:  # left
            x -= 1
        elif action == 3:  # right
            x += 1
        if self.is_valid_position(x, y):
            return (x, y)
        else:
            return state
```

**3D:** This code defines an Environment3D class representing a 3D environment for a reinforcement learning task. It initializes the environment with its dimensions, including width, height, and depth, as well as the starting point and goal location. The is_valid_position method checks whether a given position is within the boundaries of the 3D environment. Similar to the 2D version, the get_reward method returns a reward of 1.0 if the agent reaches the goal state; otherwise, it returns a reward of 0.0. The take_action method updates the agent's position based on the chosen action, allowing movement in six directions: up, down, left, right, forward, and backward, ensuring the new position remains within the 3D environment boundaries.

```python
class Environment3D:
    def __init__(self, width, height, depth, start, goal):
        self.width = width
        self.height = height
        self.depth = depth
        self.start = start
        self.goal = goal


    def is_valid_position(self, x, y, z):
        return 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth


    def get_reward(self, state):
        if state == self.goal:
            return 1.0
        else:
            return 0.0


    def take_action(self, state, action):
        x, y, z = state
        if action == 0:  # up
            y += 1
        elif action == 1:  # down
            y -= 1
        elif action == 2:  # left
            x -= 1
        elif action == 3:  # right
            x += 1
        elif action == 4:  # forward
            z += 1
        elif action == 5:  # backward
            z -= 1
        if self.is_valid_position(x, y, z):
            return (x, y, z)
        else:
            return state
```

There was an approximate disparity of ~ 65 episodes for the 2D setting, while the 3D scenario required roughly ~ 1450 episodes until stabilization. The difference between 65 episodes and 1450 episodes is approximately 22-fold, indicating that transitioning from the 2D to the 3D environment demands approximately 22 times more episodes for learning to stabilize. This stark contrast underscores the substantial increase in computational effort and time required when extending the learning framework from two dimensions to three.

___

## More About:

Its use is highly encouraged and recommended for academic and scientific research, content analysis, sentiment and speech. It is free and open, and academic use is encouraged. Its responsible use is the sole responsibility of those who adapt and manipulate the data.

___

## Author Info:

Ergon Cugler de Moraes Silva, from Brazil, mailto: <a href="contato@ergoncugler.com">contato@ergoncugler.com</a> / Master's in Public Administration and Government, Getulio Vargas Foundation (FGV).

### How to Cite it:

**SILVA, Ergon Cugler de Moraes. From Two-Dimensional to Three-Dimensional Environment with Q-Learning: Modeling Autonomous Navigation with Reinforcement Learning and no Libraries. (mar) 2024. Avaliable at: <a>https://github.com/ergoncugler/q-learning/<a>.**
