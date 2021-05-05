from DQN import Agent
import numpy as np
import gym
import matplotlib.pyplot as plt

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

if __name__ == '__main__':
    print()

    trials = []
    env = gym.make('LunarLander-v2')
    agent = Agent(gamma=0.99, epsilon=1, batch_size=64, n_actions=4,
                  eps_end=0.01, input_dims=[8], lr=0.03)
    scores, eps_history = [], []
    n_games = 500

    for i in range (n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score +=reward
            agent.store_trainsition(observation, action, reward,
                                    observation_, done)
            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])
        trials.append(i)

        print('episode', i, 'score %.2f' %score, 'average score %.2f' %avg_score,
              'epsilon %.2f' %agent.epsilon)

    mvn_score = moving_average(scores, 250)
    plt.figure(figsize=(5, 5), dpi=100)
    plt.plot(trials, scores, label='Episodic Average')
    plt.plot(mvn_score, label='Moving Average')
    plt.xlabel('Trials')
    plt.ylabel('Reward')
    plt.grid()
    plt.title('Training trial vs reward')
    plt.show()





