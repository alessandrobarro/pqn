import random
import lkh
import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from model import PQN
from tsp import generate_tsp_instance, generate_heuristic_solution, load_heuristic

LKH_path = '/Users/barro/Documents/pqn-test/LKH-3.0.9/LKH'

def prepare_inputs(tsp_instance):
    '''
    Preprare and normalizd input data for training
    '''
    coordinates = np.array([tsp_instance.nodes[node]['pos'] for node in tsp_instance.nodes])
    coordinates -= coordinates.mean(axis=0)
    coordinates /= coordinates.std(axis=0)
    return tf.convert_to_tensor(coordinates[tf.newaxis, :, :], dtype=tf.float32)  # adding 1 batch dimension

def compute_q_learning_loss(predicted_q_values, true_q_values):
    return tf.reduce_mean(tf.square(predicted_q_values - true_q_values))

def compute_ptrnet_loss(predicted_logits, true_next_indices):
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(true_next_indices, predicted_logits, from_logits=True))

def solve_tsp_with_LKH(num_cities, environment, solver_path, max_trials=10000, runs=10):
    D, cost = load_heuristic(environment)
    problem_str = "NAME : TSP\nTYPE : TSP\nDIMENSION : {}\nEDGE_WEIGHT_TYPE : EUC_2D\nNODE_COORD_SECTION\n".format(num_cities)
    for i in range(0, num_cities):
        problem_str += "{} {} {}\n".format(i+1, D[i][1], D[i][2])
    problem_str += "EOF"

    problem = lkh.LKHProblem.parse(problem_str)

    lkh_tour = lkh.solve(solver_path, problem=problem, max_trials=max_trials, runs=runs)

    # Back to 1-indexing
    seq = [city - 1 for city in lkh_tour[0]] if lkh_tour else []

    return seq

def path_cost(tsp_instance, tour_path):
    total_cost = 0
    for i in range(len(tour_path) - 1):
        total_cost += tsp_instance[tour_path[i]][tour_path[i + 1]]['weight']

    total_cost += tsp_instance[tour_path[-1]][tour_path[0]]['weight']
    return total_cost

def evaluate_metrics(model, num_cities, num_samples):
    heuristic_lengths = []
    pqn_lengths = []
    ptr_lengths = []

    for _ in range(num_samples):
        tsp_instance, modified_intance = generate_tsp_instance(num_cities)
        heuristic_path = solve_tsp_with_LKH(num_cities, tsp_instance, LKH_path)
        heuristic_cost = path_cost(tsp_instance, heuristic_path)
        heuristic_lengths.append(heuristic_cost)

        pqn_path = simulate_tsp_with_pqn(model, tsp_instance, 'pqn')
        pqn_cost = path_cost(tsp_instance, pqn_path)
        ptr_path = simulate_tsp_with_pqn(model, tsp_instance, 'ptr')
        ptr_cost = path_cost(tsp_instance, ptr_path)

        pqn_lengths.append(pqn_cost)
        ptr_lengths.append(ptr_cost)

    pqn_accuracy = np.mean([100 - ((abs((pqn - h)) / abs(h)) * 100) for pqn, h in zip(pqn_lengths, heuristic_lengths)])
    ptr_accuracy = np.mean([100 -  ((abs((ptr - h)) / abs(h)) * 100) for ptr, h in zip(ptr_lengths, heuristic_lengths)])

    return pqn_cost, pqn_accuracy, pqn_path, heuristic_path, heuristic_cost, tsp_instance, ptr_cost, ptr_accuracy, ptr_path

def simulate_tsp_with_pqn(model, tsp_instance, name):
    start_city = 0
    current_city = start_city
    visited_cities = [start_city]
    total_cost = 0
    t = 0

    while len(visited_cities) < len(tsp_instance.nodes()):
        current_sequence = tf.convert_to_tensor([visited_cities], dtype=tf.int32)
        predicted_q_values, predicted_logits, attention_scores, attention_weights, attention_scores_wo_psi, psi_inf, modified_logits = model(current_sequence)
        
        if name == 'pqn':
            next_city = int(tf.argmax(modified_logits[0], axis=-1).numpy())
        else:
            next_city = int(tf.argmax(predicted_logits[0], axis=-1).numpy())

        if next_city in visited_cities:
            possible_cities = set(range(len(tsp_instance.nodes()))) - set(visited_cities)
            if not possible_cities:
                break  # All cities have been visited
            next_city = min(possible_cities, key=lambda x: tsp_instance[current_city][x]['weight'])

        visited_cities.append(next_city)
        cost = tsp_instance[current_city][next_city]['weight']
        total_cost += cost
        current_city = next_city
        t = t + 1

    return visited_cities

def plot_tsp_solutions(tsp_instance, pqn_path, heuristic_path, ptr_path):
    pos = nx.get_node_attributes(tsp_instance, 'pos')
    plt.figure(figsize=(12, 8))

    nx.draw_networkx_nodes(tsp_instance, pos, node_shape='s', node_color='black', node_size=500)

    nx.draw_networkx_labels(tsp_instance, pos, font_color='white')

    pqn_edges = list(zip(pqn_path[:-1], pqn_path[1:])) + [(pqn_path[-1], pqn_path[0])]
    nx.draw_networkx_edges(tsp_instance, pos, edgelist=pqn_edges, edge_color='red', width=2, label='PQN Path')

    heuristic_edges = list(zip(heuristic_path[:-1], heuristic_path[1:])) + [(heuristic_path[-1], heuristic_path[0])]
    nx.draw_networkx_edges(tsp_instance, pos, edgelist=heuristic_edges, edge_color='blue', style='dashed', width=2, label='Heuristic Path')

    ptr_edges = list(zip(ptr_path[:-1], ptr_path[1:])) + [(ptr_path[-1], ptr_path[0])]
    nx.draw_networkx_edges(tsp_instance, pos, edgelist=ptr_edges, edge_color='forestgreen', width=2, label='Ptr-Net Path')

    plt.title('TSP Solutions: PQN (Red), Ptr (Green), Heuristic (Blue)', fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.show()

def plot_phi_values(psi_collected):
    plt.figure(figsize=(10, 6))
    x_positions = np.arange(len(psi_collected))

    for i, psi_values in enumerate(psi_collected):
        y_positions = psi_values
        x_values = np.full_like(y_positions, i)
        plt.scatter(x_values, y_positions, alpha=0.6, edgecolor='none', s=30)

    plt.title('Phi_exp Values Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Time Step')
    plt.ylabel('Phi Values at Each Step')
    plt.grid(True)
    plt.show()

def plot_metrics(totlosses, qlosses, ptrlosses):
    epochs = range(1, len(totlosses) + 1)
    fig, ax1 = plt.subplots()

    ax1.plot(epochs, totlosses, color='tab:red', label='Total Loss')

    ax1.plot(epochs, qlosses, color='tab:blue', label='Q-Learning Loss')

    ax1.plot(epochs, ptrlosses, color='tab:orange', label='Ptr-Net Loss')

    '''
    start_epoch = epochs[0]
    end_epoch = epochs[-1]
    '''

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='y')
    ax1.grid(True)  # Add grid lines for loss axis
    ax1.legend(loc='upper right')  # Add a legend to identify the plots

    plt.title('Losses Over Epochs', fontsize=18, fontweight='bold')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

def save_model(model, path):
    model.save_weights(path)

def load_model(model, path):
    model.load_weights(path)

def generate_training_data(num_cities, num_samples):
    training_data = []
    for _ in range(num_samples):
        tsp_normal, tsp_modified = generate_tsp_instance(num_cities)
        path = solve_tsp_with_LKH(num_cities, tsp_normal, LKH_path)

        for i in range(1, len(path)):
            current_sequence = path[:i]
            next_city = path[i]
            current_sequence_tensor = tf.convert_to_tensor(current_sequence, dtype=tf.int32)
            next_city_tensor = tf.convert_to_tensor([next_city], dtype=tf.int32)
            q_value = -tsp_normal[path[i-1]][path[i]]['weight']
            q_value_tensor = tf.convert_to_tensor([q_value], dtype=tf.float32)
            training_data.append((current_sequence_tensor, next_city_tensor, q_value_tensor, tsp_normal, tsp_modified))

    return training_data


def train(model, optimizer, training_data, epochs=100):
    avg_losses = []
    avgq_losses = []
    avgptr_losses = []
    psi_inf = []
    att = []
    att_wo_psi = []
    prev_loss = None

    print("Q trainable variables:", model.q_weights)
    print("Ptr trainable variables:", model.ptr_weights)
    


    for epoch in range(epochs):

        total_loss_epoch = 0
        qloss_epoch = 0
        ptrloss_epoch = 0
        t = 0
        modify_epochs_range = (10, 15)

        for current_sequence, next_city, q_value, tsp_normal, tsp_modified in training_data:
            if len(current_sequence.shape) == 1:
                current_sequence = tf.expand_dims(current_sequence, 0)

            
            '''
            #PERTURBATIONS
            use_modified = epoch in range(*modify_epochs_range)
            tsp_instance_to_use = tsp_modified if use_modified else tsp_normal
            '''
            


            with tf.GradientTape(persistent=True) as tape:
                predicted_q_values, predicted_logits, attention_weights, attention_scores, attention_scores_wo_psi, psi, modified_logits = model(current_sequence)
                next_city = tf.reshape(next_city, shape=(-1,))  # flattening for loss comp
                q_value = tf.reshape(q_value, shape=(-1,))

                q_loss = compute_q_learning_loss(predicted_q_values, q_value)
                ptr_loss = compute_ptrnet_loss(predicted_logits, next_city)
                total_loss = q_loss + ptr_loss

            q_gradients = tape.gradient(q_loss, model.q_weights)
            ptr_gradients = tape.gradient(ptr_loss, model.ptr_weights)

            optimizer.apply_gradients(zip(q_gradients, model.q_weights))
            optimizer.apply_gradients(zip(ptr_gradients, model.ptr_weights))

            # check nan grads
            if any(tf.reduce_any(tf.math.is_nan(grad)).numpy() for grad in q_gradients if grad is not None) or \
            any(tf.reduce_any(tf.math.is_nan(grad)).numpy() for grad in ptr_gradients if grad is not None):
                print("NaN gradients detected")
                break

            # check for weights format consistency
            if hasattr(model, 'q_weights') and isinstance(model.q_weights, list):
                q_vars = [v for sublist in model.q_weights for v in sublist if isinstance(v, tf.Variable)]
            else:
                q_vars = model.trainable_variables

            if hasattr(model, 'ptr_weights') and isinstance(model.ptr_weights, list):
                ptr_vars = [v for sublist in model.ptr_weights for v in sublist if isinstance(v, tf.Variable)]
            else:
                ptr_vars = model.trainable_variables

            q_gradients = tape.gradient(q_loss, q_vars)
            ptr_gradients = tape.gradient(ptr_loss, ptr_vars)

            optimizer.apply_gradients(zip(q_gradients, q_vars))
            optimizer.apply_gradients(zip(ptr_gradients, ptr_vars))

            total_loss_epoch += total_loss.numpy()
            qloss_epoch += q_loss.numpy()
            ptrloss_epoch += ptr_loss.numpy()
            avg_loss = total_loss_epoch / len(training_data)
            avg_q_loss = qloss_epoch / len(training_data)
            avg_ptr_loss = ptrloss_epoch / len(training_data)
            avg_losses.append(avg_loss)
            avgq_losses.append(avg_q_loss)
            avgptr_losses.append(avg_ptr_loss)
            psi_inf.append(psi)
            att.append(attention_scores)
            att_wo_psi.append(attention_scores_wo_psi)
            t = t + 1
            
            del tape  # free memory


        print(f"Epoch {epoch + 1}, Average Total Loss: {avg_loss}")


    return avg_losses, psi_inf, att, att_wo_psi, t, avgq_losses, avgptr_losses

def main():
    num_cities = 20
    num_samples = 5

    model = PQN(num_cities)
    
    model.build((1, num_cities))

    training_data = generate_training_data(num_cities, num_samples)

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.1)

    train_loss, psi_inf, att, att_wo_psi, timesteps, qloss, ptrloss = train(model, optimizer, training_data, epochs=30)
    model.save_weights('model_weights.h5')
    model.load_weights('model_weights.h5')

    plot_metrics(train_loss, qloss, ptrloss)
    plot_phi_values(psi_inf)

    pqn_cost, pqn_accuracy, pqn_path, heuristic_path, heuristic_cost, tsp_instance, ptr_cost, ptr_accuracy, ptr_path = evaluate_metrics(model, num_cities=20, num_samples=1)

    print('*-----------------------------------------------------*')
    print(f"PQN Cost: {pqn_cost}")
    print(f"Ptr Cost: {ptr_cost}")
    print(f"Heuristic Cost: {heuristic_cost}")
    print('\n')
    print(f"PQN Accuracy: {pqn_accuracy}%")
    print(f"Ptr Accuracy: {ptr_accuracy}%")
    print('\n')
    print(f"PQN Path: {pqn_path}")
    print(f"Ptr Path: {ptr_path}")
    print('\n')
    print(f"Heuristic Path: {heuristic_path}")
    print('*-----------------------------------------------------*')

    plot_tsp_solutions(tsp_instance, pqn_path, heuristic_path, ptr_path)

if __name__ == "__main__":
    main()





