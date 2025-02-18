# MOD550 - Assignment 2
# Dea Lana Asri - 277575

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import timeit as it
from mse_vanilla import mean_squared_error as vanilla_mse
from mse_numpy import mean_squared_error as numpy_mse
from sklearn.metrics import mean_squared_error as sk_mse
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Random seed for reproducibility
np.random.seed(1234)

# Assignment Point 1: Fix the code
observed = [2, 4, 6, 8]
predicted = [2.5, 3.5, 5.5, 7.5]
karg = {'observed': observed,'predicted': predicted}
factory = {'mse_vanilla' : vanilla_mse,
            'mse_numpy' : numpy_mse,
            'mse_sk' : sk_mse
            }
for talker, worker in factory.items():
    # SK method expects y_true and y_pred as an input, not dictionary
    if talker == 'mse_sk':
        exec_time = it.timeit('{worker(observed, predicted)}', globals=globals(), number=100) / 100
        mse = worker(observed, predicted)
    else:
        exec_time = it.timeit('{worker(**karg)}', globals=globals(), number=100) / 100
        mse = worker(**karg)
    print(f"Mean Squared Error, {talker} :", mse, f"Average execution time: {exec_time} seconds")

print('Task 1: Test successful')


# Assignment Point 2: Function that generate 1D oscillatory data
def generate_oscillatory_data(n_points = 200, amplitude = 2, range = [0, 3], noise = 1, random_seed=1234):
    """ Generate 1D oscillatory data.
    Parameters:
    -----------
    n_points : int, The number of points to generate.
    frequency : float, The frequency of the oscillation.
    range : tuple, The range of the data.
    noise : float, The standard deviation of the noise.
    random_seed : int, The random seed for reproducibility.

    Output:
    -------
    x : numpy.ndarray, The x values.
    y : numpy.ndarray, The y values.
    y_noise : numpy.ndarray, The y values with noise.
    """
    # Generate the data
    x = np.linspace(range[0], range[1], n_points)
    y = amplitude * np.sin(np.pi * x)
    
    # Adding noise
    noise_distribution = np.random.normal(0, noise, n_points)
    y_noise = y + noise_distribution
    return x, y, y_noise

# Generate the data
x, y, y_noise = generate_oscillatory_data()

print("Task 2: Data generated")
print(f"Truth Function: y = amplitude * sin(pi * x), {len(y)} points, range from {x[0]} to {x[-1]}, min value: {min(y):.2f}, max value: {max(y):.2f}")
print(f"Data with noise: {len(y_noise)} points, range from {x[0]} to {x[-1]}, min value: {min(y_noise):.2f}, max value: {max(y_noise):.2f}")
      
# Assignment Point 3: Clustering the data
# Prepare the data
data = np.column_stack((x, y_noise))

# Number of clusters
n_clusters = range(1, 11)
variance = []

# Lopping to cluster the data and print the information
print(f"Task 3: Clustering methond: KMeans clustering")
for i in n_clusters:
    kmeans = KMeans(n_clusters=i, random_state=1234).fit(data)
    variance.append(kmeans.inertia_)
    print(f"Number of clusters: {i}, Variance: {kmeans.inertia_}")

# Assignment Point 4, 5, 6, 7: Regression

# Reshape x and y_noise
x = x.reshape(-1, 1)
y_noise = y_noise.reshape(-1, 1)

# Linear regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x, y_noise)
y_predict = model.predict(x)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(x, y_noise, label='Data with noise')
ax.plot(x, y, label='Truth function', color='red')
ax.plot(x, y_predict, label='Linear regression', color='green')
ax.set_title('Linear regression')
ax.legend()
plt.show()

print("Task 3: Linear regression completed")

# Neural Network
# Initialize Neural Network model
nn_model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1, warm_start=True, random_state=1234)

# Set up plot
plt.ion()
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Left plot: Regression progress
ax[0].scatter(x, y_noise, label="Data with Noise", alpha=0.5)
ax[0].plot(x, y, label="Truth Function", linestyle="dashed", color="black")
line, = ax[0].plot(x, np.zeros_like(x), label="NN Prediction", color="red")
ax[0].set_title("Neural Network Regression Progress")
legend = ax[0].legend()

# Right plot: MSE and Progress
ax[1].set_title("MSE and Progress")
ax[1].set_xlabel("Iteration")
ax[1].set_xlim(0, 200)
ax[1].set_ylabel("Error / Loss")

# Initialize tracking lists
iteration_func = []
error = []
progress = []

# Lines for updating the right plot
line2, = ax[1].plot(iteration_func, error, label="MSE (to truth)", color="blue")
line3, = ax[1].plot(iteration_func, progress, label="Progress (truth not known)", color="green")
ax[1].legend()

# Initialize iterations
iterations = 200
i = 1

while i <= iterations:
    nn_model.max_iter = i
    nn_model.fit(x, y_noise.ravel())  
    y_nn_pred = nn_model.predict(x)  

    # Calculate error and progress
    iteration_func.append(i)
    mse = mean_squared_error(y, y_nn_pred) 
    error.append(mse)  
    progress.append(nn_model.loss_)
    
    # Update label for NN prediction with current iteration number
    line.set_ydata(y_nn_pred)  
    line2.set_xdata(iteration_func)  
    line2.set_ydata(error)  
    line3.set_xdata(iteration_func)  
    line3.set_ydata(progress)  

    # Rescale axes
    ax[1].relim()
    ax[1].autoscale_view()

    # Update NN prediction label
    line.set_label(f"NN Prediction (Iteration {i})")
    legend.remove()
    legend = ax[0].legend()

    plt.draw()
    plt.pause(0.1)  # Pause to create animation effect
    
    i += 10

plt.ioff()  # Turn off interactive mode
plt.show()
print("Task 3: Neural Network completed")

# Physics-Informed Neural Network
import torch
import torch.nn as nn
import torch.optim as optim

# Define the PINN model
class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

# Define physics-based loss function
def physics_loss(model, x):
    x.requires_grad = True
    y_pred = model(x)
    y_pred_x = torch.autograd.grad(y_pred, x, torch.ones_like(y_pred), create_graph=True)[0]
    y_pred_xx = torch.autograd.grad(y_pred_x, x, torch.ones_like(y_pred_x), create_graph=True)[0]
    loss = torch.mean((y_pred_xx + y_pred) ** 2)
    return loss

torch.manual_seed(1234)

# Convert to tensors
x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_noise_tensor = torch.tensor(y_noise, dtype=torch.float32).view(-1, 1)

# Initialize model and optimizer
pinn_model = PINN()
optimiser = optim.Adam(pinn_model.parameters(), lr=1e-3)

# Define boundary points for the boundary loss
# The solution at t=0 is known and will be enforced as a constraint
t_boundary = torch.tensor(0.).view(-1, 1).requires_grad_(True)

# Set up animated plots
plt.ion()
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Left plot: Regression progress
ax[0].scatter(x, y_noise, label="Data with Noise", alpha=0.5)
ax[0].plot(x, y, label="Truth Function", linestyle="dashed", color="black")
line, = ax[0].plot(x, np.zeros_like(x), label="PINN Prediction", color="red")
ax[0].set_title("Physics-Informed Neural Network Regression Progress")
legend = ax[0].legend()

# Right plot: Loss evolution
ax[1].set_title("Loss Evolution")
ax[1].set_xlabel("Iteration")
ax[1].set_xlim(0, 5000)
ax[1].set_ylabel("Loss")

# Initialize tracking lists
iteration_func = []
loss_list = []
progress = []

# Loss line
line2, = ax[1].plot([], [], label="Loss (to truth)", color="blue")
line3, = ax[1].plot([], [], label="Progress (truth not known)", color="green")
ax[1].legend()

# Training loop
iterations = 5000
i = 1
while i <= iterations+1:
    optimiser.zero_grad()

    # Compute Data Loss (Fitting the noisy observations)
    u_data_pred = pinn_model(x_tensor)
    data_loss = torch.mean((u_data_pred - y_noise_tensor) ** 2)

    # Compute Boundary Loss
    u = pinn_model(t_boundary)
    boundary_loss = (torch.squeeze(u) - 1) ** 2

    # Compute the derivative of u with respect to time at t=0
    dudt = torch.autograd.grad(u, t_boundary, torch.ones_like(u), create_graph=True)[0]
    loss2 = (torch.squeeze(dudt) - 0) ** 2  # Enforce u'(0) = 0 (initial velocity)

    # Compute Physics Loss
    physic_loss = physics_loss(pinn_model, x_tensor)


    # Weight for each loss
    lambda_data = 1.0
    lambda_physics = 0.01 
    lambda_boundary = 0.1

    # Weighted sum of the losses
    loss = lambda_data * data_loss + lambda_physics * physic_loss + lambda_boundary * boundary_loss
    mse = mean_squared_error(y, u_data_pred.detach().numpy())

    loss.backward()
    optimiser.step()

    iteration_func.append(i)
    loss_list.append(mse)
    progress.append(loss.item())
    

    y_pinn_pred = pinn_model(x_tensor).detach().numpy()
    line.set_ydata(y_pinn_pred)
    line2.set_xdata(iteration_func)
    line2.set_ydata(loss_list)

    line3.set_xdata(iteration_func)
    line3.set_ydata(progress)
        
    # Rescale loss plot
    ax[1].relim()
    ax[1].autoscale_view()

    # Update iteration label
    line.set_label(f"PINN Prediction (Iteration {i})")
    legend.remove()
    legend = ax[0].legend()

    plt.draw()
    plt.pause(0.05)
    i += 10

plt.ioff()
plt.show()
print("Task 3: Physics-Informed Neural Network Training Completed")

# Assignment Point 8: Run reinforcement learning code

# Copying Reinforcement Learning Code
# GridWorld Environment
class GridWorld:
    """GridWorld environment with obstacles and a goal.
    The agent starts at the top-left corner and has to reach the bottom-right corner.
    The agent receives a reward of -1 at each step, a reward of -0.01 at each step in an obstacle, and a reward of 1 at the goal.

    Args:
        size (int): The size of the grid.
        num_obstacles (int): The number of obstacles in the grid.

    Attributes:
        size (int): The size of the grid.
        num_obstacles (int): The number of obstacles in the grid.
        obstacles (list): The list of obstacles in the grid.
        state_space (numpy.ndarray): The state space of the grid.
        state (tuple): The current state of the agent.
        goal (tuple): The goal state of the agent.

    Methods:
        generate_obstacles: Generate the obstacles in the grid.
        step: Take a step in the environment.
        reset: Reset the environment.
    """
    def __init__(self, size=5, num_obstacles=5):
        self.size = size
        self.num_obstacles = num_obstacles
        self.obstacles = [(0, 4), (4, 3), (1, 3), (1, 0), (3, 2)]
        self.state_space = np.zeros((self.size, self.size))
        self.state = (0, 0)
        self.goal = (self.size-1, self.size-1)

    def step(self, action):
        """
        Take a step in the environment.
        The agent takes a step in the environment based on the action it chooses.

        Args:
            action (int): The action the agent takes.
                0: up
                1: right
                2: down
                3: left

        Returns:
            state (tuple): The new state of the agent.
            reward (float): The reward the agent receives.
            done (bool): Whether the episode is done or not.
        """
        x, y = self.state

        if action == 0:  # up
            x = max(0, x-1)
        elif action == 1:  # right
            y = min(self.size-1, y+1)
        elif action == 2:  # down
            x = min(self.size-1, x+1)
        elif action == 3:  # left
            y = max(0, y-1)
        self.state = (x, y)
        if self.state in self.obstacles:
         #   self.state = (0, 0)
            return self.state, -1, True
        if self.state == self.goal:
            return self.state, 1, True
        return self.state, -0.1, False

    def reset(self):
        """
        Reset the environment.
        The agent is placed back at the top-left corner of the grid.

        Args:
            None

        Returns:
            state (tuple): The new state of the agent.
        """
        self.state = (0, 0)
        return self.state

# Q-Learning
class QLearning:
    """
    Q-Learning agent for the GridWorld environment.

    Args:
        env (GridWorld): The GridWorld environment.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.
        episodes (int): The number of episodes to train the agent.

    Attributes:
        env (GridWorld): The GridWorld environment.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        epsilon (float): The exploration rate.
        episodes (int): The number of episodes to train the agent.
        q_table (numpy.ndarray): The Q-table for the agent.

    Methods:
        choose_action: Choose an action for the agent to take.
        update_q_table: Update the Q-table based on the agent's experience.
        train: Train the agent in the environment.
        save_q_table: Save the Q-table to a file.
        load_q_table: Load the Q-table from a file.
    """
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=0.1, episodes=10):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episodes = episodes
        self.q_table = np.zeros((self.env.size, self.env.size, 4))

    def choose_action(self, state):
        """
        Choose an action for the agent to take.
        The agent chooses an action based on the epsilon-greedy policy.

        Args:
            state (tuple): The current state of the agent.

        Returns:
            action (int): The action the agent takes.
                0: up
                1: right
                2: down
                3: left
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice([0, 1, 2, 3])  # exploration
        else:
            return np.argmax(self.q_table[state])  # exploitation

    def update_q_table(self, state, action, reward, new_state):
        """
        Update the Q-table based on the agent's experience.
        The Q-table is updated based on the Q-learning update rule.

        Args:
            state (tuple): The current state of the agent.
            action (int): The action the agent takes.
            reward (float): The reward the agent receives.
            new_state (tuple): The new state of the agent.

        Returns:
            None
        """
        self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + \
            self.alpha * (reward + self.gamma * np.max(self.q_table[new_state]))

    def train(self):
        """
        Train the agent in the environment.
        The agent is trained in the environment for a number of episodes.
        The agent's experience is stored and returned.

        Args:
            None

        Returns:
            rewards (list): The rewards the agent receives at each step.
            states (list): The states the agent visits at each step.
            starts (list): The start of each new episode.
            steps_per_episode (list): The number of steps the agent takes in each episode.
        """
        rewards = []
        states = []  # Store states at each step
        starts = []  # Store the start of each new episode
        steps_per_episode = []  # Store the number of steps per episode
        steps = 0  # Initialize the step counter outside the episode loop
        episode = 0
        #print(self.q_table)
        while episode < self.episodes:
            state = self.env.reset()
            total_reward = 0
            done = False
            #print(f"Episode {episode+1}")
            #print(self.q_table)
            while not done:
                action = self.choose_action(state)
                new_state, reward, done = self.env.step(action)
                self.update_q_table(state, action, reward, new_state)
                state = new_state
                total_reward += reward
                states.append(state)  # Store state
                steps += 1  # Increment the step counter
                if done and state == self.env.goal:  # Check if the agent has reached the goal
                    starts.append(len(states))  # Store the start of the new episode
                    rewards.append(total_reward)
                    steps_per_episode.append(steps)  # Store the number of steps for this episode
                    steps = 0  # Reset the step counter
                    episode += 1
        return rewards, states, starts, steps_per_episode
    

# Plot the number of iterations needed to converge as a function of the learning rate.
learning_rates = np.linspace(0.1, 1, 20)
iterations = []

for alpha in learning_rates:
    env = GridWorld()
    agent = QLearning(env, alpha=alpha, episodes=100)
    rewards, states, starts, steps_per_episode = agent.train()

    # Define convergence as when the agent consistently reaches the goal in a small number of steps
    moving_avg = np.convolve(steps_per_episode, np.ones(5)/5, mode='valid')
    # Since the goal can be reached in 8 steps, we consider convergence to be when the moving average is <= 12
    convergence_episode = np.argmax(moving_avg <= 12) if np.any(moving_avg <= 12) else len(steps_per_episode)
    iterations.append(convergence_episode)

# Plot Learning Rate vs. Convergence Episodes
plt.figure(figsize=(8, 6))
plt.plot(learning_rates, iterations, marker='o', linestyle='-')
plt.xlabel("Learning Rate")
plt.ylabel("Iterations to Converge")
plt.title("Effect of Learning Rate on Convergence")
plt.grid(True)
plt.show()