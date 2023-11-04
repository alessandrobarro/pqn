import torch
import torch.nn as nn
import torch.nn.functional as F

class PtrNet(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, graph_size):
        super(PtrNet, self).__init__()

        self.graph_size = graph_size
        
        self.embedding_dim = embedding_dim
        self.embedding = nn.Linear(input_dim, embedding_dim)
        self.node_embedding = nn.Embedding(graph_size, embedding_dim)
        self.action_embedding = nn.Embedding(graph_size, embedding_dim)
        
        # Encoder and Decoder LSTMs
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        
        # Linear layers for attention mechanism
        self.W_ref = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, state):
        # Ensure input state is a tensor
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Extract R from the state
        R = state[:, :self.graph_size]
        mask = (R != -1).float().unsqueeze(-1)
        R = R.clone()
        R[R == -1] = 0
        
        # Apply node embeddings
        embedded_R = self.node_embedding(R.long())
        
        # Nullify embeddings of padding nodes using the mask
        embedded_R = embedded_R * mask
        
        # Pass R through the encoder
        encoder_outputs, (hidden, cell) = self.encoder(embedded_R)
        
        batch_size = embedded_R.size(0)

        # Initialize decoder input as zero
        decoder_input = torch.zeros(batch_size, 1, self.embedding_dim, device=state.device)

        # Initial hidden states for the decoder are the reshaped states of the encoder
        decoder_hidden = (hidden, cell)
        
        # Placeholder for the attention weights
        attention_weights = []
        
        # Placeholder for the chosen actions/indices
        actions = []
        
        # Initialize mask (1)
        mask = torch.ones(batch_size, embedded_R.size(1), device=state.device)
        
        # Decoding steps
        for t in range(self.graph_size):

            # Pass through the decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            # Calculate attention weights
            ref = self.W_ref(encoder_outputs)
            q = self.W_q(decoder_output)
            energies = self.v(torch.tanh(ref + q)).squeeze(-1)
            probs = F.softmax(energies, dim=1)

            # Apply mask
            probs_clone = probs.squeeze()
            masked_probs = probs_clone * mask
            masked_probs = masked_probs / masked_probs.sum(1, keepdim=True)
            
            attention_weights.append(masked_probs.unsqueeze(1))

            # Get the most probable next input
            _, topi = masked_probs.topk(1)

            # Safeguard against out-of-range indices
            topi = topi % self.graph_size

            # Update the mask
            mask = mask.scatter(1, topi, 0)
                                    
            decoder_input = self.action_embedding(topi.squeeze(1).detach()).unsqueeze(1)
            
            actions.append(topi)
        
        actions = torch.stack(actions, dim=1)
        attention_weights = torch.cat(attention_weights, dim=1)
        
        return actions, attention_weights

    
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim
        
    def forward(self, state_action_pairs):
        x = F.relu(self.fc1(state_action_pairs))
        q_values = self.fc2(x)

        return q_values.view(-1, self.output_dim)

class PQN(nn.Module):
    def __init__(self, input_dim, embedding_dim, graph_size, hidden_dim=128, lambda_value=5):
        super(PQN, self).__init__()
        self.lambda_value = lambda_value
        self.graph_size = graph_size
        self.ptr_net = PtrNet(input_dim, embedding_dim, hidden_dim, graph_size)

        # Modify the input_dim for QNetwork initialization
        self.q_network = QNetwork(input_dim * 2 + 1 + embedding_dim, 1, embedding_dim)
        self.embedding = nn.Embedding(graph_size, embedding_dim)

    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        _, probabilities = self.ptr_net(state)
        
        # Initialize the mask to ones
        mask = torch.ones_like(probabilities)
        
        # Hold the top-k action indices for each batch item
        topk_actions = []
        
        # Iterate lambda times to get top-k actions
        for _ in range(self.lambda_value):
            masked_probs = probabilities * mask
            top_action = torch.argmax(masked_probs, dim=1, keepdim=True)
            topk_actions.append(top_action)
            
        
        action_indices = torch.cat(topk_actions, dim=1)

        # Convert action_indices to embeddings
        action_embeddings = self.embedding(action_indices)  # [batch_size, lambda, graph_size, embedding_dim]
        
        # Sum over the graph_size dimension to reduce it
        action_embeddings = action_embeddings.sum(dim=2)  # [batch_size, lambda, embedding_dim]

        # Create the lambda state-action pairs
        state_repeated = state.unsqueeze(1).repeat(1, self.lambda_value, 1)
        state_action_pairs = torch.cat((state_repeated, action_embeddings), dim=-1)

        # Flatten the batch and lambda dimensions together for QNetwork
        state_action_pairs_flattened = state_action_pairs.view(-1, state.size(-1) + self.embedding.embedding_dim)
        q_values = self.q_network(state_action_pairs_flattened)

        # Reshape Q values to separate batch and lambda dimensions
        q_values = q_values.view(-1, self.lambda_value)

        # Select the action with the highest Q-value
        best_action_idx = torch.argmax(q_values, dim=1)
        best_action = action_indices[torch.arange(action_indices.size(0)), best_action_idx]
        return best_action, probabilities