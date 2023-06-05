import numpy as np
import pprint

gamma = 0.9
alpha = 0.1
reward = np.array([
    [0, 1, 0, 1, 0, 0, 0, 0, 0], # from 0 to ...
    [1, 0, 1, 0, 1, 0, 0, 0, 0], # from 1 to ...
    [0, 1, 0, 0, 0, 0, 0, 0, 0], # from 2 to ...
    [1, 0, 0, 0, 0, 0, 1, 0, 0], # from 3 to ...
    [0, 1, 0, 0, 0, 1, 0, 1, 0], # from 4 to ...
    [0, 0, 0, 0, 1, 0, 0, 0, 100], # from 5 to ...
    [0, 0, 0, 1, 0, 0, 0, 1, 0], # from 6 to ...
    [0, 0, 0, 0, 1, 0, 1, 0, 0], # from 7 to ...
    [0, 0, 0, 0, 0, 1, 0, 0, 0], # from 8 to ...
])

Q = np.array(np.zeros([9, 9]))

for i in range(1000):
    p_state = np.random.randint(0, len(reward[0])) # choose 1 int from 0 to 9
    n_actions = [] # pick up available actions
    for j in range(len(reward[p_state])):
        if reward[p_state, j] > 0:
            n_actions.append(j)
    n_state = np.random.choice(n_actions) # choose an action from available actions

    Q[p_state, n_state] = (1-alpha) * Q[p_state, n_state] + \
        alpha * (reward[p_state, n_state] + gamma * Q[n_state, np.argmax(Q[n_state,])])

    if i % 100 == 0:
        print('=== Epoch: %d ===' % (i))
        pprint.pprint(Q, width=1000)

def shortest_path(start):
    path = [start]
    while path[-1] != 8:
        path.append(np.argmax(Q[path[-1],]))
        
if __name__ == '__main__':
    shortest_path(0)
