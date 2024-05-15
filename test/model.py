import numpy as np
import tensorflow as tf

def phi_function(q_values, t, T, alpha=2.0):
    phi_values = q_values * np.exp(alpha / (T - t + 1))
    return phi_values

'''
def psi_function(q_values, t, T, alpha=2.0):
    psi_values = q_values * t
    return psi_values
'''

class PQN(tf.keras.Model):
    def __init__(self, num_cities, lstm_units=256, T=100):
        super(PQN, self).__init__()
        self.time = 0
        self.T = T

        # Layers
        self.lstm = tf.keras.layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.attention = tf.keras.layers.Attention(use_scale=True)
        self.dense_q = tf.keras.layers.Dense(num_cities, activation='linear')
        self.dense_logits = tf.keras.layers.Dense(num_cities, activation=None)
        self.embedding = tf.keras.layers.Embedding(input_dim=num_cities, output_dim=2)

        # Trainable weights
        self.q_weights = []
        self.ptr_weights = []

    def build(self, input_shape):
        super(PQN, self).build(input_shape)
        self.q_weights = self.dense_q.trainable_variables
        self.ptr_weights = self.dense_logits.trainable_variables + self.lstm.trainable_variables

    def call(self, inputs):
        if self.time % self.T == 0:
            self.time = 0
        self.time = self.time + 1

        if len(inputs.shape) == 1:
            inputs = tf.expand_dims(inputs, 0)

        city_embeddings = self.embedding(inputs)
        lstm_output, state_h, state_c = self.lstm(city_embeddings)
        query = tf.expand_dims(state_h, 1)
        context_vector, attention_weights = self.attention([query, lstm_output], return_attention_scores=True)

        q_values = self.dense_q(context_vector)
        logits = self.dense_logits(context_vector)  # raw u_ta
        phi_values = phi_function(q_values, self.time, self.T)
        modified_logits = tf.multiply(logits, phi_values) # u_ta * Phi(Q)
        attention_scores = tf.nn.softmax(modified_logits, axis=-1)
        attention_scores_wo_psi = tf.nn.softmax(logits, axis=-1) # for comparison only
        psi_inf = tf.norm(phi_values, ord=np.inf, axis=None)
        
        return q_values, logits, attention_scores, attention_weights, attention_scores_wo_psi, psi_inf, modified_logits
