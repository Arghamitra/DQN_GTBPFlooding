import numpy as np
from GTBPFloodingParallel import *
from param import *

class Environment:
    def __init__(self, trial):
        self.previous_LLR = np.array(n)
        self.trial = trial
        self._step = 0
        self.prim_valus()


    def prim_valus(self):
        np.random.seed(self.trial)
        # x = a binary vector of length n, representing the status of individuals in the population
        x = np.array([0 for i in range(n)])
        # I = sorted index set of infected people
        I = np.random.choice(n, k, replace=False)
        I = np.sort(I)
        self.I = I
        for i in I:
            x[i] = 1

        # y = a binary vector of length m, representing the true test results
        y = Boolean_Measurements(H, x)
        # y_hat = a binary vector of length m, representing the noisy version of test results
        self.y_hat = Binary_Symmetric_Channel(y, delta, self.trial)

    def DQN_state_maker(self, cn_connection, LLR):
        state = 0
        for cnctn in cn_connection:
            state += LLR[cnctn]
        return state

    def reset(self):
        self._step = 0
        states = []
        self.allc_to_v_messages = np.array([[[0.5, 0.5] for i in range(len(N_v[v]))] for v in range(n)],
                                      dtype=object)
        self.allv_to_c_messages = np.array([[[1 - q, q] for i in range(len(N_c[c]))] for c in range(m)],
                                      dtype=object)
        Pre_LLR = Compute_LLR(n, q, N_v, self.allc_to_v_messages)
        self.previous_LLR = Pre_LLR

        # calculation state: eqn 6 (midterm report)
        for c in range(m):
            cn_connection = N_c[c]
            state = self.DQN_state_maker(cn_connection, Pre_LLR)
            states.append(state)
        observation = states
        return observation

    def done_flag(self, LLR):
        if self._step > no_iter:
            return True
        if L1_norm(LLR, self.previous_LLR) <= 0.001:
            return True
        else:
            return False


    def reward(self, LLR, crct_cn_cnctn):
        nw_arr = np.abs(LLR - self.previous_LLR)
        diff_arr = []
        for cnctn in crct_cn_cnctn:
            diff_arr.append(nw_arr[cnctn])
        reward = max(diff_arr)
        return reward

    def new_iter_LLR(self):
        done = False
        for c in range(m):
            c_val = self.y_hat[c]
            Temp_N_c = N_c[c]
            for v in Temp_N_c:
                Temp_mesages = c_to_v_messages(c_val, Temp_N_c, v, self.allv_to_c_messages[c], delta)
                c_index = np.where(N_v[v] == c)
                self.allc_to_v_messages[v][c_index[0][0]][0] = Temp_mesages[0]
                self.allc_to_v_messages[v][c_index[0][0]][1] = Temp_mesages[1]

        LLR = Compute_LLR(n, q, N_v, self.allc_to_v_messages)
        if L1_norm(LLR, self.previous_LLR) <= 0.001:
            done = True
        if self._step > no_iter:
            done = True
        else:
            for v in range(n):
                Temp_N_v = N_v[v]
                for c in Temp_N_v:
                    Temp_mesages = v_to_c_message(Temp_N_v, c, self.allc_to_v_messages[v], q)
                    v_index = np.where(N_c[c] == v)
                    self.allv_to_c_messages[c][v_index[0][0]][0] = Temp_mesages[0]
                    self.allv_to_c_messages[c][v_index[0][0]][1] = Temp_mesages[1]

        return LLR, done



    def step(self, action):

        self._step +=1
        states = []
        c_val = self.y_hat[action]
        Temp_N_c = N_c[action]
        for v in Temp_N_c:
            Temp_N_v = N_v[v]
            Temp_mesages = c_to_v_messages(c_val, Temp_N_c, v, self.allv_to_c_messages[action], delta)
            c_index = np.where(N_v[v] == action)
            self.allc_to_v_messages[v][c_index[0][0]][0] = Temp_mesages[0]
            self.allc_to_v_messages[v][c_index[0][0]][1] = Temp_mesages[1]

            for c in Temp_N_v:
                Temp_mesages = v_to_c_message(Temp_N_v, c, self.allc_to_v_messages[v], q)
                v_index = np.where(N_c[c] == v)
                self.allv_to_c_messages[c][v_index[0][0]][0] = Temp_mesages[0]
                self.allv_to_c_messages[c][v_index[0][0]][1] = Temp_mesages[1]
        try:
            LLR, done = self.new_iter_LLR()
        except:
            f =3

        # calculation state: eqn 6 (midterm report)
        for c in range(m):
            cn_connection = N_c[c]
            state = self.DQN_state_maker(cn_connection, LLR)
            states.append(state)
        observation = states

        #reward
        crct_cn_cnctn = N_c[action]
        reward = self.reward(LLR, crct_cn_cnctn)

        info = 0

        self.previous_LLR = LLR
        return observation, reward, done, info


#****OLD********OLD********OLD********OLD********OLD********OLD********OLD********OLD********OLD********OLD****


    def BP_Flooding_M(self, y_hat, n, m, k, N_c, N_v, q, delta, no_iter):

        x_hat1 = self.previous_LLR >= 0
        x_hat1 = x_hat1.astype(int)
        I_hat1 = np.where(x_hat1 == 1)[0]

        x_hat2 = np.array([0 for i in range(n)])
        sorted_LLR_indices = np.argsort(self.previous_LLR)[::-1]
        x_hat2[sorted_LLR_indices[:k]] = 1
        I_hat2 = np.where(x_hat2 == 1)[0]

        return I_hat1, I_hat2



    def CodeRunner(self, trial):


        # BP algorithm with flooding
        # I_hat1 = Estimate of I when k is unknown
        # I_hat2 = Estimate of I when k is known
        # I_hat1, I_hat2 = DQN_env(y_hat, n, m, k, N_c, N_v, q, delta, no_iter)
        I_hat1, I_hat2 = self.BP_Flooding_M(self.y_hat, n, m, k, N_c, N_v, q, delta, no_iter)

        # SSR1 = an indicator variable representing whether BP algorithm recovered x correctly or not (when k is unknown)
        SSR1 = int(np.array_equal(I_hat1, self.I))
        # SSR2 = an indicator variable representing whether BP algorithm recovered x correctly or not (when k is known)
        SSR2 = int(np.array_equal(I_hat2, self.I))
        # FNR1 = false negative rate (when k is unknown)
        FNR1 = len(np.setdiff1d(self.I, I_hat1)) / k
        # FPR1 = false positive rate (when k is unknown)
        FPR1 = len(np.setdiff1d(I_hat1, self.I)) / (n - k)
        # FNR2 = false negative rate (when k is known)
        FNR2 = len(np.setdiff1d(self.I, I_hat2)) / k
        # FPR2 = false positive rate (when k is known)
        FPR2 = len(np.setdiff1d(I_hat2, self.I)) / (n - k)

        return [SSR1, SSR2, FNR1, FPR1, FNR2, FPR2]

    def show_result(self, results):
        SSR1 = np.mean(np.array([results[i][0] for i in range(no_trials)]))
        SSR2 = np.mean(np.array([results[i][1] for i in range(no_trials)]))
        FNR1 = np.mean(np.array([results[i][2] for i in range(no_trials)]))
        FPR1 = np.mean(np.array([results[i][3] for i in range(no_trials)]))
        FNR2 = np.mean(np.array([results[i][4] for i in range(no_trials)]))
        FPR2 = np.mean(np.array([results[i][5] for i in range(no_trials)]))

        print('Success probability for unknown k = {}'.format(SSR1))
        print('Success probability for known k = {}'.format(SSR2))
        print('Unknown k: False negative rate = {}, False positive rate = {}'.format(FNR1, FPR1))
        print('Known k: False negative rate = {}, False positive rate = {}'.format(FNR2, FPR2))


