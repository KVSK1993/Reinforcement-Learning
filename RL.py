import numpy as np
import MDP
import random
import matplotlib.pyplot as plt


class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state])
        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]
    
    
    '''def selectAction(self,curState,Q_s_a,epsilon,temperature):
        if epsilon == 0 and temperature == 0:
            return np.argmax(Q_s_a[:,curState])
        
        randomProb = np.random.uniform(0.0,1.0)
        if randomProb <= epsilon:
            return random.randint(0,self.mdp.nActions-1)#choose action randomly
        else:#Boltzman Exploration
            Q = Q_s_a[:,curState]/temperature
            nQ = np.exp(Q)
            sumQ = np.sum(nQ)
            Qt = nQ/sumQ
            return np.argmax(Qt)'''
        
    def selectAction(self,curState,Q_s_a,epsilon,temperature):
        if epsilon > 0:
            randomProb = np.random.uniform(0.0,1.0)
            if randomProb <= epsilon:
                return random.randint(0,self.mdp.nActions-1)#choose action randomly
            else:
                return np.argmax(Q_s_a[:,curState])#greedy selection
        elif temperature > 0:
            Q = Q_s_a[:,curState]/temperature
            nQ = np.exp(Q)
            sumQ = np.sum(nQ)
            Qt = nQ/sumQ
            return np.argmax(Qt)
        else:
            return np.argmax(Q_s_a[:,curState])
        
    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with 
        probabilty epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        Q = initialQ
        # policy = np.zeros(self.mdp.nStates,int)
        intState = 0
        curState = 0
        
        N = np.zeros([self.mdp.nActions,self.mdp.nStates])
        
        cummulativeRewardArray = np.zeros(nEpisodes)
        for i in range(0,nEpisodes):
            rewardTotal = 0
            curState = intState
            for j in np.arange(nSteps):
        #1, Select and execute 
                #print(i,j)
                action = RL.selectAction(self,curState,Q,epsilon,temperature)
        #2,Observing s' and r
                reward,nextState = RL.sampleRewardAndNextState(self,curState,action)
                N[action][curState] =  N[action][curState] + 1
                alpha = 1/N[action][curState]
                QCurrentStateAction = Q[action][curState]
                x = max(Q[:,nextState])
                QNextStateAction = reward + self.mdp.discount*(x)
                Q[action][curState] = QCurrentStateAction + alpha*(QNextStateAction - QCurrentStateAction)
                rewardTotal += pow(self.mdp.discount,j)*reward
                curState = nextState
            cummulativeRewardArray[i] = rewardTotal
                
            
            #N = np.zeros([self.mdp.nActions,self.mdp.nStates])# check to do or not
        #cummulativeRewardArray = cummulativeRewardArray
        policy = np.argmax(Q,axis = 0)
        return [Q,policy,cummulativeRewardArray]    