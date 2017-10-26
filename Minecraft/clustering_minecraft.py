import gym
import gym_minecraft
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import pickle
import random
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier



def make_clusters(env_name, n_clusters):
    # import clustering_minecraft as cm
    # returned_data, labels, cluster_centers = cm.make_clusters('MinecraftBasic-v0', 4)
    # from scipy.misc import toimage
    # toimage(estimator.cluster_centers_[3].reshape((84,84,3))).show()

    env = gym.make(env_name)
    with open('./game_clustering.xml', 'r') as myfile:
        game_xml = myfile.read()
    env.init(start_minecraft=True, allowDiscreteMovement=True)
    VALID_ACTIONS = 2

    data = []
    moves = {0: 0,  # move forward
             1: 8,  # turn left
             2: 9  # turn right
             }

    x_vals = map(lambda x: x + 0.5, [i for i in range(1,10)] + [i for i in range(12,21)]
                 + [i for i in range(23,32)] + [i for i in range(34,43)])
    z_vals = map(lambda x: x + 0.5, [i for i in range(1,10)])

    for episode in xrange(50):
        game_xml = re.sub(r'x=\"\d+.\d+\" z=\"\d+.\d+\"', 'x="{}" z="{}"'.format(random.choice(x_vals),
                                                                                random.choice(z_vals)), game_xml)
        env.load_mission_xml(game_xml)
        state = env.reset()
        done = False
        step_count = 0
        while not done:
            step_count += 1
            action = moves[random.randint(0, VALID_ACTIONS)]
            next_state, reward, done, _ = env.step(action)
            data.append(next_state.flatten())
    env.close()

    data = np.array(data)

    estimator = KMeans(n_clusters=n_clusters)
    estimator.fit(data)
    cluster_centers = estimator.cluster_centers_
    labels = estimator.labels_
    labels = labels.astype(np.int32)
    # for i in xrange(len(data)):
    #    if data[i][0] >= 0.5:
    #        print 'X Position:'
    #        print data[i][0]
    #        print 'Reward:'
    #        print data[i][2]
    #        print 'Cluster:'
    #        print labels[i]
    #        print ""

    try:
        os.stat('Minecraft_clusters_' + str(n_clusters))
    except:
        os.mkdir('Minecraft_clusters_' + str(n_clusters))

    returned_data = zip(data[:, 0], data[:, 1])

    with open('Minecraft_clusters_' + str(n_clusters) + '/data', 'w') as data_file:
        pickle.dump(returned_data, data_file)
    with open('Minecraft_clusters_' + str(n_clusters) + '/labels', 'w') as labels_file:
        pickle.dump(labels, labels_file)
    with open('Minecraft_clusters_' + str(n_clusters) + '/cluster_centers', 'w') as cluster_centers_file:
        pickle.dump(cluster_centers, cluster_centers_file)
    with open('Minecraft_clusters_' + str(n_clusters) + '/kmeans', 'w') as kmeans_file:
        pickle.dump(estimator, kmeans_file)

    return returned_data, labels, cluster_centers


def get_cluster_fn(env_name='MountainCar-v0', n_clusters=10, extra_bit=True, load_from_dir=True):
    if load_from_dir:
        with open('Minecraft_clusters_' + str(n_clusters) + '/data', 'rb') as data_file:
            data = pickle.load(data_file)
        with open('Minecraft_clusters_' + str(n_clusters) + '/labels', 'rb') as labels_file:
            labels = pickle.load(labels_file)
        with open('Minecraft_clusters_' + str(n_clusters) + '/cluster_centers', 'rb') as cluster_centers_file:
            cluster_centers = pickle.load(cluster_centers_file)
        with open('Minecraft_clusters_' + str(n_clusters) + '/kmeans', 'rb') as kmeans_file:
            estimator = pickle.load(kmeans_file)

    else:
        data, labels, cluster_centers = make_clusters(env_name, n_clusters)
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(data, labels)
    # Create one-hot representation of the clusters.
    clusters_one_hot = [np.zeros(n_clusters) for i in xrange(n_clusters)]
    for i in xrange(len(clusters_one_hot)):
        clusters_one_hot[i][i] = 1

    ratio = 0.5

    def check_cluster(data_point, cluster_index, original_point=None):
        # print "Check cluster function:"
        # print cluster_index
        if not extra_bit or original_point is None:
            return cluster_index == neigh.predict(data_point)[0]
        else:
            distance_to_boundary = euclidean_distance(data_point, original_point)
            distance_to_center = euclidean_distance(data_point, cluster_centers[cluster_index])
            return np.float(distance_to_center) / np.maximum(distance_to_boundary, np.exp(-10)) <= ratio


    def identify_cluster(data_point, original_point):
        cluster_index = neigh.predict(data_point)[0]
        if extra_bit:
            cluster_one_hot = np.zeros(n_clusters + 1)
        else:
            cluster_one_hot = np.zeros(n_clusters)
        cluster_one_hot[cluster_index] = 1

        if extra_bit:
            # Add bit that represents whether agent is on boundary or in center of cluster
            if original_point is not None:
                distance_to_boundary = euclidean_distance(data_point, original_point)
                distance_to_center = euclidean_distance(data_point, cluster_centers[cluster_index])
                if np.float(distance_to_center) / np.maximum(distance_to_boundary, np.exp(-10)) <= ratio:
                    cluster_one_hot[-1] = 1

        return cluster_one_hot

    return identify_cluster, check_cluster, n_clusters, np.array(clusters_one_hot)


def euclidean_distance(point1, point2):
    point1 = np.squeeze(point1)
    point2 = np.squeeze(point2)
    return np.sqrt(np.square(point1[0] - point2[0]) + np.square(point1[1] - point2[1]))
