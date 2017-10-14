import gym
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import random


def make_clusters(env_name, n_clusters):
    env = gym.make(env_name)
    env.reset()
    VALID_ACTIONS = list(range(env.action_space.n))

    data = []
    goal_x_positions = []

    for episode in xrange(100):
        state = env.reset()
        done = False
        step_count = 0
        while not done:
            step_count += 1
            action = random.randint(0, len(VALID_ACTIONS) - 1)
            next_state, reward, done, _ = env.step(action)
            if reward == 1:
                reward = 100
            state = next_state
            state_with_reward = state[:]
            state_with_reward = np.append(state_with_reward, reward)
            data.append(state_with_reward)
            if done and step_count < 200:
                goal_x_positions.append(np.squeeze(state))

    data = np.array(data)
    x_pos_normalized = (data[:, 0] - np.mean(data[:, 0])) / np.std(data[:, 0])
    velocity_normalized = (data[:, 1] - np.mean(data[:, 1])) / np.std(data[:, 1])
    reward_normalized = (data[:, 2] - np.mean(data[:, 2])) / np.std(data[:, 2])
    data_normalized = zip(x_pos_normalized, velocity_normalized, reward_normalized)
    # data_normalized = zip(x_pos_normalized, velocity_normalized)

    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(data_normalized)

    labels = estimator.labels_

    # for i in xrange(len(data)):
    #    if data[i][0] >= 0.5:
    #        print 'X Position:'
    #        print data[i][0]
    #        print 'Reward:'
    #        print data[i][2]
    #        print 'Cluster:'
    #        print labels[i]
    #        print ""

    labels = labels.astype(np.int32)
    colors = ['red', 'green', 'blue', 'orange', 'yellow']

    fig, ax = plt.subplots()
    for i in xrange(n_clusters):
        label = i
        color = colors[i]
        indices_of_labels = np.where(labels==label)
        # print indices_of_labels
        # print data[indices_of_labels,0][0]
        ax.scatter(data[indices_of_labels,0][0], data[indices_of_labels,1][0], c=color,
            label=int(label), alpha=0.5)


    # for i in xrange(len(data[:,0])):
    #    print i
    #    ax.scatter(data[i,0], data[i,1], c=labels[i], label=int(labels[i]),
    #        alpha=0.3)
    ax.legend()
    plt.xlabel('X Position')
    plt.ylabel('Velocity')
    plt.savefig('Clusters.png')

    returned_data = zip(data[:, 0], data[:, 1])
    return returned_data, labels

def get_cluster_fn(env_name='MountainCar-v0', n_clusters=5):
    data, labels = make_clusters(env_name, n_clusters)
    neigh = KNeighborsClassifier(n_neighbors=9)
    neigh.fit(data, labels)
    # Create one-hot representation of the clusters.
    clusters_one_hot = [np.zeros(n_clusters) for i in xrange(n_clusters)]
    for i in xrange(len(clusters_one_hot)):
        clusters_one_hot[i][i] = 1

    def check_cluster(data_point, cluster_index):
        # print "Check cluster function:"
        # print cluster_index
        # print neigh.predict(data_point)[0]
        return cluster_index == neigh.predict(data_point)[0]

    def identify_cluster(data_point):
        cluster_index = neigh.predict(data_point)[0]
        cluster_one_hot = np.zeros(n_clusters)
        cluster_one_hot[cluster_index] = 1
        return cluster_one_hot

    return identify_cluster, check_cluster, n_clusters, np.array(clusters_one_hot)
