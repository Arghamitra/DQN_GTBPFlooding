from DQN import Agent
import time
from environment import Environment
from param import *
import matplotlib.pyplot as plt
import dill

gamma = 0.99
epsilon = 1
batch_size= 16
lr=0.03
print('EPISODE', no_trials)
print('gamma=', gamma, 'starting epsilon=',
      epsilon, 'batch size', batch_size, 'lr', lr)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def training():
    print('___________TRAINING______________')
    agents = []
    for idx_done in range(no_iter):
        agent = Agent(gamma=gamma, epsilon=epsilon, batch_size=batch_size, n_actions=m,
                  eps_end=0.01, input_dims=[m], lr=lr)
        agents.append(agent)
    scores, eps_history = [], []
    start_time = time.time()
    results = []
    trials = []

    for trial in range(no_trials):
        env = Environment(trial,  status = 'train')
        score = 0
        done = False
        observation = env.reset()
        i = 0

        for idx_done in range (no_iter):
            i +=1
            action = agents[idx_done].choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score +=reward
            agents[idx_done].store_trainsition(observation, action, reward,
                                    observation_, done)
            agents[idx_done].learn()
            observation = observation_
            #print('epsilon %.2f' % agents[idx_done].epsilon)

        result = env.CodeRunner(trial)
        results.append(result)
        scores.append(score)
        #eps_history.append(agent.epsilon)
        avg_score = np.mean(scores)
        trials.append(trial)
        print('episode', trial, 'score %.2f' %score, 'average score %.2f' %avg_score,
              'epsilon %.4f' %agents[idx_done].epsilon)


    mvn_score = moving_average(scores, 250)
    plt.figure(figsize=(5, 5), dpi=100)
    plt.plot(trials, scores, label='Episodic Average')
    plt.plot(mvn_score, label='Moving Average')
    plt.xlabel('Trials')
    plt.ylabel('Reward')
    plt.grid()
    plt.title('Training trial vs reward')
    trimester = time.strftime("_%Y_%m_%d-%H__%M_%S")
    plt.savefig('./pic/TR_DQN_' + str(no_trials) + '_' + (trimester) + '.png')

    print("--- %s TRAINING seconds ---" % (time.time() - start_time))
    env.show_result(results)
    handle_agents(opt = 'save', agents = agents)


def handle_agents(opt='save', agents = None):
    path = 'all_agents.pkl'
    if opt == 'save':
        with open(path, 'wb') as f:
            dill.dump(agents, f)
    elif opt == 'load':
        with open (path, 'rb') as f:
            agents = dill.load(f)
            return agents



def testing():
    print('___________TESTING_____________')
    agents = handle_agents(opt='load')
    scores, eps_history = [], []
    start_time = time.time()
    results = []
    trials = []
    for trial in range(no_trials):
        env = Environment(trial,  status = 'test')
        score = 0
        done = False
        observation = env.reset()
        i = 0
        for idx_done in range(no_iter):
            i += 1
            action = agents[idx_done].choose_action(observation, epsilon = 0)
            observation_, reward, done, info = env.step(action)
            score += reward
            observation = observation_

        result = env.CodeRunner(trial)
        results.append(result)
        scores.append(score)
        avg_score = np.mean(scores)
        trials.append(trial)
        print('episode', trial, 'score %.2f' % score, 'average score %.2f' % avg_score,
              'epsilon %.2f' % agents[idx_done].epsilon)

    mvn_score = moving_average(scores, 250)
    plt.figure(figsize=(5, 5), dpi=100)
    plt.plot(trials, scores, label='Episodic Average')
    plt.plot(mvn_score, label='Moving Average')
    plt.xlabel('Trials')
    plt.ylabel('Reward')
    plt.grid()
    plt.title('Testing trial vs reward')
    trimester = time.strftime("_%Y_%m_%d-%H__%M_%S")
    plt.savefig('./pic/TS_DQN_' + str(no_trials) + '_' + (trimester) + '.png')

    print("--- %s TESTING seconds ---" % (time.time() - start_time))
    env.show_result(results)

if __name__ == '__main__':
    training()
    testing()













