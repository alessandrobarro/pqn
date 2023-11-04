# Modules
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from env import OPEnvironment
from model import PtrNet, PQN

# Hyperparameters
GRAPH_SIZE = 20
H = 128
LR = 1e-3
GAMMA = 0.99
EPISODES = {20: 20, 100: 100}

# Helper functions
def visualize_routes(graph, route1, route2, title='Routes Comparison'):

    if not hasattr(graph, 'coords'): # Circular coordinates
        theta = np.linspace(0, 2 * np.pi, graph.V, endpoint=False)
        graph.coords = np.column_stack((np.cos(theta), np.sin(theta)))

    fig, ax = plt.subplots(figsize=(10, 8))

    for v, coord in enumerate(graph.coords): # Vertices
        ax.plot(coord[0], coord[1], 'ko')  # 'ko' makes the point black and circular
        ax.text(coord[0], coord[1], str(v), color='white', ha='center', va='center')

    for i in range(len(route1) - 1): # Edges 1
        v1, v2 = route1[i], route1[i + 1]
        ax.plot(*zip(graph.coords[v1], graph.coords[v2]), 'r-')

    for i in range(len(route2) - 1): # Edges 2
        v1, v2 = route2[i], route2[i + 1]
        ax.plot(*zip(graph.coords[v1], graph.coords[v2]), 'b-')

    ax.set_title(title)
    ax.axis('equal')
    ax.axis('off')

    plt.show()

def add_coordinates_to_graph(graph):
    theta = np.linspace(0, 2 * np.pi, graph.V, endpoint=False)
    graph.coords = np.column_stack((np.cos(theta), np.sin(theta)))

# Training
def train_ptrnet(ptrnet, env, optimizer, episodes):
    torch.autograd.set_detect_anomaly(True)

    ptrnet_rewards = []
    ptrnet_losses = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        total_loss = 0
        timestep = 0

        print("-"*60)
        print(f"EPISODE: {episode}")
        
        while not done and timestep < GRAPH_SIZE:

            # Predict the next action using PtrNet
            actions, attention_weights = ptrnet(state)
            action = torch.argmax(attention_weights[0, timestep]).item()

            action_sequence = '→'.join(map(str, actions.squeeze().tolist()))
            print(f"[{action_sequence}]")

            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            total_reward += reward

            # Calculate the cross-entropy loss for the chosen action
            log_probs = F.log_softmax(attention_weights, dim=-1)
            loss = F.nll_loss(log_probs[0, timestep].unsqueeze(0), torch.tensor([action]))
            total_loss += loss.item()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            timestep += 1
            
            state = next_state
        
        ptrnet_rewards.append(total_reward)
        ptrnet_losses.append(total_loss)

        print(f"Reward: {total_reward}")
        print(f"Loss: {total_loss}")
    
    return ptrnet_rewards, ptrnet_losses

def train_pqn(pqn, env, optimizer, episodes, epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=200):
    pqn_rewards = []
    pqn_losses = []
    
    for episode in range(episodes):
        state = env.reset()

        # Convert state to tensor
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        done = False
        total_reward = 0
        total_loss = 0

        print("-"*60)
        print(f"EPISODE: {episode}")
        
        while not done:
            # Decay epsilon
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-1. * episode / epsilon_decay)

            # Epsilon-greedy action selection
            actions, probabilities = pqn(state)
            action_sequence = '→'.join(map(str, actions.squeeze().tolist()))
            print(f"[{action_sequence}]")

            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.randint(0, GRAPH_SIZE)
            else:
                # Select the action with the highest Q-value
                best_action_idx = torch.argmax(actions, dim=1)
                action = actions[0, best_action_idx].item()

            # Take a step in the environment
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Obtain the Q-value for the current state and action
            _, next_probabilities = pqn(next_state)
            next_q_values, _ = torch.max(next_probabilities, dim=1)
            expected_q_value = reward + GAMMA * next_q_values
            
            # Calculate the Q-learning loss
            _, current_probabilities = pqn(state)
            current_q_value = current_probabilities[0, action]
            
            loss = F.mse_loss(current_q_value, expected_q_value.detach())
            total_loss += loss.item()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            state = next_state
        
        pqn_rewards.append(total_reward)
        pqn_losses.append(total_loss)

        print(f"Reward: {total_reward}")
        print(f"Loss: {total_loss}")
    
    return pqn_rewards, pqn_losses

# Evaluation
def evaluate(model, env, is_ptrnet=True):
    state = env.reset()
    done = False
    total_reward = 0
    total_loss = 0
    timestep = 0
    actions_taken = set()

    if is_ptrnet:
        actions, attention_weights = model(state)
        for action in actions.squeeze().tolist():
            if action in actions_taken:
                break
            actions_taken.add(action)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Compute the loss for this step
            log_probs = F.log_softmax(attention_weights, dim=-1)
            loss = F.nll_loss(log_probs[0, timestep].unsqueeze(0), torch.tensor([action]))
            total_loss += loss.item()
            timestep += 1

            if done:
                break
            state = next_state
    else:
        while not done:
            actions, probabilities = model(state)
            sorted_actions = torch.argsort(actions, descending=True)
            for idx in range(sorted_actions.shape[1]):
                action = sorted_actions[0, idx].item()
                if action not in actions_taken:
                    break
            actions_taken.add(action)
            
            next_state, reward, done, _ = env.step(action)
            total_reward += reward

            # Compute the loss for this step
            _, next_probabilities = model(next_state)
            next_q_values, _ = torch.max(next_probabilities, dim=1)
            expected_q_value = reward + GAMMA * next_q_values

            _, current_probabilities = model(state)
            current_q_value = current_probabilities[0, action]

            loss = F.mse_loss(current_q_value.unsqueeze(0), expected_q_value.detach())
            total_loss += loss.item()

            state = next_state

    return total_reward, total_loss, list(actions_taken)

if __name__ == "__main__":
    for GRAPH_SIZE, episodes in EPISODES.items():
        print(f"Running experiments for graph size {GRAPH_SIZE}...")
        print("="*60)

        # Initialize environments
        train_env = OPEnvironment(GRAPH_SIZE)
        eval_env = OPEnvironment(GRAPH_SIZE)
        test_env = OPEnvironment(GRAPH_SIZE)

        # Generate coordinates for the graph
        add_coordinates_to_graph(train_env.graph)
        add_coordinates_to_graph(eval_env.graph)
        add_coordinates_to_graph(test_env.graph)

        # Set lambda and other hyperparameters based on GRAPH_SIZE
        L_s = GRAPH_SIZE // 2
        LAMBDA = L_s // 2
        O_s = L_s
        STATE_SIZE = GRAPH_SIZE * 2 + 1

        # Initialize models and optimizers
        ptrnet = PtrNet(input_dim=GRAPH_SIZE, embedding_dim=H, hidden_dim=H, graph_size=GRAPH_SIZE)
        pqn = PQN(input_dim=GRAPH_SIZE, embedding_dim=H, graph_size=GRAPH_SIZE, lambda_value=LAMBDA)

        ptrnet_optimizer = torch.optim.Adam(ptrnet.parameters(), lr=LR)
        pqn_optimizer = torch.optim.Adam(pqn.parameters(), lr=LR)

        # Train loop
        print("-"*60)
        print("PtrNet:")
        ptrnet_rewards, ptrnet_losses = train_ptrnet(ptrnet, train_env, ptrnet_optimizer, episodes)
        print("\nPQN:")
        pqn_rewards, pqn_losses = train_pqn(pqn, train_env, pqn_optimizer, episodes)
        print("-"*60)
        
        # Training
        print("\nTRAINING RESULTS")

        ptrnet_train_reward, ptrnet_train_loss, ptrnet_train_policy = evaluate(ptrnet, train_env, is_ptrnet=True)
        print("PtrNet -> Reward:", ptrnet_train_reward, "CE Loss:", ptrnet_train_loss)
        print("Actions:", '→'.join(map(str, ptrnet_train_policy)))

        pqn_train_reward, pqn_train_loss, pqn_train_policy = evaluate(pqn, train_env, is_ptrnet=False)
        print("\nPQN -> Reward:", pqn_train_reward, "Q-MSE Loss:", pqn_train_loss)
        print("Actions:", '→'.join(map(str, pqn_train_policy)))

        print("="*60)
        
        # Evaluation
        print("\nEVALUATION RESULTS")
        
        ptrnet_eval_reward, ptrnet_eval_loss, ptrnet_eval_policy = evaluate(ptrnet, eval_env, is_ptrnet=True)
        print("PtrNet -> Reward:", ptrnet_eval_reward, "Loss:", ptrnet_eval_loss)
        print("Actions:", '→'.join(map(str, ptrnet_eval_policy)))

        pqn_eval_reward, pqn_eval_loss, pqn_eval_policy = evaluate(pqn, eval_env, is_ptrnet=False)
        print("\nPQN -> Reward:", pqn_eval_reward, "Q-MSE Loss:", pqn_eval_loss)
        print("Actions:", '→'.join(map(str, pqn_eval_policy)))

        visualize_routes(eval_env.graph, ptrnet_eval_policy, pqn_eval_policy, title='Evaluation Route')

        print("="*60)

        # Test
        print("\nTEST RESULTS")

        ptrnet_test_reward, ptrnet_test_loss, ptrnet_test_policy = evaluate(ptrnet, test_env, is_ptrnet=True)
        print("\nPtrNet -> Reward:", ptrnet_test_reward, "Loss:", ptrnet_test_loss)
        print("Actions:", '→'.join(map(str, pqn_eval_policy)))

        pqn_test_reward, pqn_test_loss, pqn_test_policy = evaluate(pqn, test_env, is_ptrnet=False)
        print("\nPQN -> Reward:", pqn_test_reward, "Q-MSE Loss:", pqn_test_loss)
        print("Actions:", '→'.join(map(str, pqn_eval_policy)))

        visualize_routes(test_env.graph, ptrnet_test_policy, pqn_test_policy, title='Test Route')
