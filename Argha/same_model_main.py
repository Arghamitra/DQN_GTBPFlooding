from DQN import Agent
import time
from environment import Environment
from param import *
import matplotlib.pyplot as plt
import pickle


#no_trials = 1
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def training():
    print('___________TRAINING______________')
    agent = Agent(gamma=0.99, epsilon=1, batch_size=64, n_actions=m,
                  eps_end=0.01, input_dims=[m], lr=0.03)
    #agent = agent.cuda()
    scores, eps_history = [], []
    start_time = time.time()
    results = []
    trials = []

    for trial in range(no_trials):
        env = Environment(trial, status = 'train')
        score = 0
        done = False
        observation = env.reset()
        i = 0
        while not done:
            i +=1
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score +=reward
            agent.store_trainsition(observation, action, reward,
                                    observation_, done)
            agent.learn()
            observation = observation_
            #print('episode', trial, 'iteration', i, 'reward %.2f' % reward,
                  #'epsilon %.2f' % agent.epsilon)

        result = env.CodeRunner(trial)
        results.append(result)
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        trials.append(trial)
        print('episode', trial, 'score %.2f' %score, 'average score %.2f' %avg_score,
              'epsilon %.4f' %agent.epsilon)

    mvn_score = moving_average(scores, 250)

    plt.figure(figsize=(5, 5), dpi=100)
    plt.plot(trials, scores, label='Episodic Average')
    plt.plot(mvn_score, label='Moving Average')
    plt.xlabel('Trials')
    plt.ylabel('Reward')
    plt.grid()
    plt.title('Training trial vs reward')
    trimester = time.strftime("_%Y_%m_%d-%H__%M_%S")
    plt.savefig('./pic/TrnS_DQN_' + str(no_trials) + '_' + (trimester) + '.png')

    print("--- %s seconds ---" % (time.time() - start_time))
    env.show_result(results)
    handle_agents(opt='save', agents=agent)

def handle_agents(opt='save', agents = None):
    path = 'all_agents.pkl'
    if opt == 'save':
        with open(path, 'wb') as f:
            pickle.dump(agents, f)
    elif opt == 'load':
        with open (path, 'rb') as f:
            agents = pickle.load(f)
            return agents

def testing():
    print('___________TESTING_____________')
    agent = handle_agents(opt='load')

    scores, eps_history = [], []
    start_time = time.time()
    results = []
    trials = []

    for trial in range(no_trials):
        env = Environment(trial, status='test')
        score = 0
        done = False
        observation = env.reset()
        i = 0
        while not done:
            i += 1
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            observation = observation_

        result = env.CodeRunner(trial)
        results.append(result)
        scores.append(score)
        avg_score = np.mean(scores[-100:])
        trials.append(trial)
        print('episode', trial, 'score %.2f' % score, 'average score %.2f' % avg_score,
              'epsilon %.4f' % agent.epsilon)

    mvn_score = moving_average(scores, 250)

    plt.figure(figsize=(5, 5), dpi=100)
    plt.plot(trials, scores, label='Episodic Average')
    plt.plot(mvn_score, label='Moving Average')
    plt.xlabel('Trials')
    plt.ylabel('Reward')
    plt.grid()
    plt.title('Testing trial vs reward')
    trimester = time.strftime("_%Y_%m_%d-%H__%M_%S")
    plt.savefig('./pic/TstS_DQN_' + str(no_trials) + '_' + (trimester) + '.png')

    print("--- %s seconds ---" % (time.time() - start_time))
    env.show_result(results)


if __name__ == '__main__':
    training()
    testing()