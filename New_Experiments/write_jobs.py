
clusters_arr = [5, 10, 20]
array = [True, False]

for c in clusters_arr:
    for a1 in array:
        for a2 in array:
            for a3 in array:
                for a4 in array:
                    for a5 in array:
                        print 'python train_dqn.py --agent_type=h_dqn --n_clusters=' + str(c) + ' --use_extra_travel_penalty=' + str(a1) + ' --use_extra_bit=' + str(a2) + ' --use_intrinsic_timeout=' + str(a3) + ' --use_controller_dqn=' + str(a4) + ' --use_memory=' +str(a5)