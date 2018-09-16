import numpy as np
import MDP
import random
import math

class RL2:
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

    def sampleSoftmaxPolicy(self,policyParams,state):
        '''Procedure to sample an action from stochastic policy
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))])
        This function should be called by reinforce() to selection actions

        Inputs:
        policyParams -- parameters of a softmax policy (|A|x|S| array)
        state -- current state

        Outputs: 
        action -- sampled action
        '''

        # temporary value to ensure that the code compiles until this
        # function is coded
        #action = 0
        num = np.exp(policyParams[:,state])
        den = np.sum(num)
        num = num/den
        
        action = np.argmax(np.random.multinomial(1,num))
        
        return action

    def modelBasedRL(self,s0,defaultT,initialR,nEpisodes,nSteps,epsilon=0):
        '''Model-based Reinforcement Learning with epsilon greedy 
        exploration.  This function should use value iteration,
        policy iteration or modified policy iteration to update the policy at each step

        Inputs:
        s0 -- initial state
        defaultT -- default transition function when a state-action pair has not been vsited
        initialR -- initial estimate of the reward function
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random

        Outputs: 
        V -- final value function
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.amax(initialR,axis = 0)#np.zeros(self.mdp.nStates)
        policy = np.zeros(self.mdp.nStates,int)
        R = initialR #A*S
        T = defaultT
        stateActionCnt = np.zeros((self.mdp.nActions,self.mdp.nStates))
        stateActionStateCnt = np.zeros((self.mdp.nActions,self.mdp.nStates,self.mdp.nStates))
        intState = s0
        curState = s0
        #trialCnt = 100
        cumRewardArray = np.zeros(nEpisodes)
        for episode in range(0,nEpisodes):
            totalG = 0
            #for trial in range(trialCnt):
            curState = intState
            for timeSteps in range(0,nSteps):
                randomProb = np.random.uniform(0.0,1.0)
                if randomProb <= epsilon:
                    action = random.randint(0,self.mdp.nActions-1)#choose action randomly
                else:
                    pol = np.argmax(np.add(R,self.mdp.discount*np.dot(T,V)),axis = 0)#greedy selection
                    action = pol[curState]
                    
                reward, nextState = self.sampleRewardAndNextState(curState,action) # observe s' and r
                stateActionCnt[action][curState] += 1 #update n(s,a)
                stateActionStateCnt[action][curState][nextState] += 1 #update n(s,a,s')
                T[action][curState] = stateActionStateCnt[action][curState]/stateActionCnt[action][curState] #update Transistion function for all s'
                R[action][curState] = (reward + (stateActionCnt[action][curState] - 1)*R[action][curState])/stateActionCnt[action][curState] #update Reward for s'
                totalG += math.pow(self.mdp.discount,timeSteps)*reward
                mdp = MDP.MDP(T,R,self.mdp.discount)
                #V = np.amax(np.add(R,self.mdp.discount*np.dot(T,V)),axis = 0)
                [V,nIterations,epsilon] = mdp.valueIteration(initialV=np.zeros(mdp.nStates))    
                #policy = mdp.extractPolicy(V)
                #V = mdp.evaluatePolicy(policy)
                curState = nextState
            cumRewardArray[episode] = totalG
            #print("avgCumRewardArray",cumRewardArray[episode])
        policy = mdp.extractPolicy(V)
        return [V,policy,cumRewardArray]    

    def epsilonGreedyBandit(self,nIterations):
        '''Epsilon greedy algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        
        
        #trials = 1000
        avgRewardArray = np.zeros(nIterations)
        empiricalMeans = np.zeros(self.mdp.nActions)
        countAction = np.zeros(self.mdp.nActions)
        for i in range(1,nIterations+1):
        #itr = 1
        #rewardTotal = 0
        #
        #
        #while itr <= trials:
            
            randomProb = np.random.uniform(0.0,1.0)
            epsilon = 1/i
            if randomProb <= epsilon:
                action = random.randint(0,self.mdp.nActions-1)#choose action randomly
            else:
                action = np.argmax(empiricalMeans)#greedy selection
            #maintain count how many times each action is getting called
            countAction[action] += 1 
            curState = self.mdp.nStates - 1#we have only a single state
            reward, nextState = self.sampleRewardAndNextState(curState,action)
            empiricalMeans[action] = (empiricalMeans[action]*(countAction[action] - 1) + reward)/countAction[action]
            
            #rewardTotal += reward
            avgRewardArray[i-1] = reward
            #print(empiricalMeans)
        
        return avgRewardArray

    def thompsonSamplingBandit(self,prior,nIterations,k=1):
        '''Thompson sampling algorithm for Bernoulli bandits (assume no discount factor)

        Inputs:
        prior -- initial beta distribution over the average reward of each arm (|A|x2 matrix such that prior[a,0] is the alpha hyperparameter for arm a and prior[a,1] is the beta hyperparameter for arm a)  
        nIterations -- # of arms that are pulled
        k -- # of sampled average rewards

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        
        
        
        #trial = 1000
        avgRewardArray = np.zeros(nIterations)
        #j = 0
        #for j in range(nIterations):
            #rewardTotal = 0
        #empiricalMeans = np.zeros(self.mdp.nActions)
        n = 0
        while n < nIterations:
            empiricalMeans = np.zeros(self.mdp.nActions)
            for i in range(0,k):
                empiricalMeans = np.add(empiricalMeans,np.random.beta(prior[:,0],prior[:,1]))
            empiricalMeans = empiricalMeans/k
                #print("sample",rewardSample)
            action = np.argmax(empiricalMeans)
            curState = self.mdp.nStates - 1
            reward, nextState = self.sampleRewardAndNextState(curState,action)
            if reward == 1:
                prior[action][0] += 1
            else:
                prior[action][1] += 1
            avgRewardArray[n] = reward
            n = n+1
            #print(j)
            #avgRewardArray[j] = rewardTotal/trial
        return avgRewardArray

    def UCBbandit(self,nIterations):
        '''Upper confidence bound algorithm for bandits (assume no discount factor)

        Inputs:
        nIterations -- # of arms that are pulled

        Outputs: 
        empiricalMeans -- empirical average of rewards for each arm (array of |A| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        #trials= 1000
        
        avgRewardArray = np.zeros(nIterations)
        
        
        #for i in range(0,nIterations):
        empiricalMeans = np.zeros(self.mdp.nActions)      
        #rewardTotal = 0
        n = 1
        countAction = np.ones(self.mdp.nActions)
        while n <= nIterations:
            
            x = 2*math.log(n)
            y = np.sqrt(x/countAction)
            y = np.add(y, empiricalMeans)
            action = np.argmax(y)
            countAction[action] += 1 
            curState = self.mdp.nStates - 1
            reward, nextState = self.sampleRewardAndNextState(curState,action)
            empiricalMeans[action] = (empiricalMeans[action]*(countAction[action] - 1) + reward)/countAction[action]
            
            #rewardTotal += reward
            avgRewardArray[n-1] = reward
            n += 1
        #print(countAction)
        return avgRewardArray

    def reinforce(self,s0,initialPolicyParams,nEpisodes,nSteps):
        '''reinforce algorithm.  Learn a stochastic policy of the form
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))]).
        This function should call the function sampleSoftmaxPolicy(policyParams,state) to select actions

        Inputs:
        s0 -- initial state
        initialPolicyParams -- parameters of the initial policy (array of |A|x|S| entries)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0)
        nSteps -- # of steps per episode

        Outputs: 
        policyParams -- parameters of the final policy (array of |A|x|S| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        
        initialState = s0
        policyParams = initialPolicyParams
        countAction = np.zeros((self.mdp.nActions,self.mdp.nStates))  
        totalG = 0
        #trialCnt = 100
        avgCumRewardArray = np.zeros(nEpisodes)
        for n in range(nEpisodes):
            totalG = 0
            #for trial in range(trialCnt):
            curState = initialState
            sampleEpisode = list()
            discountedReward = np.zeros(nSteps)
             
            for steps in range(nSteps):
                action = self.sampleSoftmaxPolicy(policyParams,curState)
                reward, nextState = self.sampleRewardAndNextState(curState,action)
                tup = (curState,action,reward)
                sampleEpisode.append(tup)
                
                discountedReward[steps] = math.pow(self.mdp.discount,steps)*reward
                #totalG += discountedReward[steps]
                
                curState = nextState
            totalG += np.sum(discountedReward) 
            for steps in range(nSteps):
                GnArray = discountedReward[steps:]/math.pow(self.mdp.discount,steps)
                Gn = np.sum(GnArray)
                currentStep = sampleEpisode[steps]
                state1 = currentStep[0]
                action1 = currentStep[1]
                countAction[action1][state1] +=1
                alpha = 0.1#1/countAction[action1][state1]
                gradientMatrix = self.computeGradient(policyParams[:,state1],state1,action1)
                policyParams = np.add(policyParams , alpha*math.pow(self.mdp.discount,steps)*Gn*gradientMatrix)
            
            avgG = totalG
        #print("avgG",avgG)
            avgCumRewardArray[n] = avgG
        #policyParams = np.zeros((self.mdp.nActions,self.mdp.nStates))
            
        return policyParams,avgCumRewardArray    
    
    def computeGradient(self,paramVector,state,action):
        
        gradientMatrix = np.zeros((self.mdp.nActions,self.mdp.nStates))
        paramVector = paramVector - np.max(paramVector)
        expParamVector = np.exp(paramVector)
        den = np.sum(expParamVector)
        expParamVector = expParamVector/den
        for act in range(self.mdp.nActions):
            if act == action:
                gradientMatrix[act][state] = 1 - expParamVector[act]
            else:
                gradientMatrix[act][state] = -expParamVector[act]
        '''otherAction = 0 if action == 1 else 1
        theta1 = paramVector[action]
        theta2 = paramVector[otherAction]
        x = 1 /(1 + math.exp(theta1 - theta2))
        
        gradientMatrix[action][state] = x
        gradientMatrix[otherAction][state] = -x'''
        
        
        return gradientMatrix