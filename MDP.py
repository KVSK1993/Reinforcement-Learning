import numpy as np

class MDP:
    '''A simple MDP class.  It includes the following members'''

    def __init__(self,T,R,discount):
        '''Constructor for the MDP class

        Inputs:
        T -- Transition function: |A| x |S| x |S'| array
        R -- Reward function: |A| x |S| array
        discount -- discount factor: scalar in [0,1)

        The constructor verifies that the inputs are valid and sets
        corresponding variables in a MDP object'''

        assert T.ndim == 3, "Invalid transition function: it should have 3 dimensions"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions,self.nStates,self.nStates), "Invalid transition function: it has dimensionality " + repr(T.shape) + ", but it should be (nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "Invalid transition function: some transition probability does not equal 1"
        self.T = T
        assert R.ndim == 2, "Invalid reward function: it should have 2 dimensions" 
        assert R.shape == (self.nActions,self.nStates), "Invalid reward function: it has dimensionality " + repr(R.shape) + ", but it should be (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "Invalid discount factor: it should be in [0,1)"
        self.discount = discount
        
    def valueIteration(self,initialV,nIterations=np.inf,tolerance=0.01):
        '''Value iteration procedure
        V <-- max_a R^a + gamma T^a V

        Inputs:
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''
        
        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.nStates)
        iterId = 0
        epsilon = 0
        
        V_n1 = np.amax(self.R,axis = 0)
        #print("Vn1",V_n1.shape)
        V_n = np.zeros(self.nStates)
        #print("Vn",V_n.shape)
        #temp = np.zeros(self.nStates)
        #epsilon = np.linalg.norm(np.subtract(V_n,V_n1),np.inf)
        
        while iterId < nIterations :
            
            V_n = np.amax(np.add(self.R,self.discount*np.dot(self.T,V_n1)),axis = 0)
            iterId = iterId +1
            #print("itr", iterId, V_n)
            V = V_n1
            V_n1 = V_n
            V_n = V
            epsilon = np.linalg.norm(np.subtract(V_n,V_n1),np.inf)
            if epsilon < tolerance:
                break
        return V_n,iterId,epsilon

    def extractPolicy(self,V):
        '''Procedure to extract a policy from a value function
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- Value function: array of |S| entries

        Output:
        policy -- Policy: array of |S| entries'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policy = np.zeros(self.nStates)
        policy = np.argmax(np.add(self.R,self.discount*np.dot(self.T,V)),axis = 0)
        return policy 

    def evaluatePolicy(self,policy):
        '''Evaluate a policy by solving a system of linear equations
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- Policy: array of |S| entries

        Ouput:
        V -- Value function: array of |S| entries'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        V = np.zeros(self.nStates)
        newR = np.zeros(self.nStates)
        for s in range(self.nStates):
            action = policy[s]
            newR[s] = self.R[action][s]
            T_action = (self.T[action][s]).reshape(1,self.nStates)
            if s == 0:
                newT = T_action
            else:
                newT = np.concatenate((newT,T_action),axis = 0)
        #(I-GT)V = R  || AX = B
        IMatrix = np.identity(self.nStates)
        
        leftSide = np.subtract(IMatrix , self.discount*newT)
        V = np.linalg.solve(leftSide,newR)
        return V
        
    def policyIteration(self,initialPolicy,nIterations=np.inf):
        '''Policy iteration procedure: alternate between policy
        evaluation (solve V^pi = R^pi + gamma T^pi V^pi) and policy
        improvement (pi <-- argmax_a R^a + gamma T^a V^pi).

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        nIterations -- limit on # of iterations: scalar (default: inf)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policyT = np.zeros(self.nStates)
        V_n = np.zeros(self.nStates)
        iterId = 0
        
        policyN = initialPolicy
        policyN1 = np.full(self.nStates,-1)
        while not np.array_equal(policyN, policyN1):
            V_n = MDP.evaluatePolicy(self,policyN) # policy evaluation
            policyN1 = MDP.extractPolicy(self,V_n) # policy improve
            iterId = iterId+1 # iteration count
            policyT = policyN1
            policyN1 = policyN
            policyN = policyT
            print("itr",iterId, V_n)
        return policyN,V_n,iterId
            
    def evaluatePolicyPartially(self,policy,initialV,nIterations=np.inf,tolerance=0.01):
        '''Partial policy evaluation:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- Policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nIterations -- limit on the # of iterations: scalar (default: infinity)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        V -- Value function: array of |S| entries
        iterId -- # of iterations performed: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        tempV = np.zeros(self.nStates)
        V_n1 = initialV
        V_n = np.full(self.nStates,-1)
        iterId = 0
        epsilon = 0
        newR = np.zeros(self.nStates)
        
        #getting transition function and reward acc to policy
        for s in range(self.nStates):
            action = policy[s]
            newR[s] = self.R[action][s]
            T_action = (self.T[action][s]).reshape(1,self.nStates)
            if s == 0:
                newT = T_action
            else:
                newT = np.concatenate((newT,T_action),axis = 0)
                
        epsilon = np.linalg.norm(np.subtract(V_n,V_n1),np.inf)       
        for i in np.arange(nIterations):
            if epsilon > tolerance:
                V_n = np.add(newR,self.discount*np.dot(newT,V_n1))
                tempV = V_n
                V_n = V_n1
                V_n1 = tempV
                iterId = iterId +1
                epsilon = np.linalg.norm(np.subtract(V_n,V_n1),np.inf)
            else:
                break
        return [V_n1,iterId,epsilon]

    def modifiedPolicyIteration(self,initialPolicy,initialV,nEvalIterations=5,nIterations=np.inf,tolerance=0.01):
        '''Modified policy iteration procedure: alternate between
        partial policy evaluation (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        and policy improvement (pi <-- argmax_a R^a + gamma T^a V^pi)

        Inputs:
        initialPolicy -- Initial policy: array of |S| entries
        initialV -- Initial value function: array of |S| entries
        nEvalIterations -- limit on # of iterations to be performed in each partial policy evaluation: scalar (default: 5)
        nIterations -- limit on # of iterations to be performed in modified policy iteration: scalar (default: inf)
        tolerance -- threshold on ||V^n-V^n+1||_inf: scalar (default: 0.01)

        Outputs: 
        policy -- Policy: array of |S| entries
        V -- Value function: array of |S| entries
        iterId -- # of iterations peformed by modified policy iteration: scalar
        epsilon -- ||V^n-V^n+1||_inf: scalar'''

        # temporary values to ensure that the code compiles until this
        # function is coded
        
        tempV = np.zeros(self.nStates)
        iterId = 0
        epsilon = 0
        
        policyN = initialPolicy
        V_n = initialV
        V_n1 = np.full(self.nStates,-1)
        #epsilon = np.linalg.norm(np.subtract(V_n,V_n1),np.inf)
        
        while iterId < nIterations:
            #policy evaluation
            V_n, itr,epi = MDP.evaluatePolicyPartially(self,policyN,V_n,nEvalIterations)
            #policy improvement
            policyN = MDP.extractPolicy(self,V_n)
            #value improvement
            V_n1 = np.amax(np.add(self.R,self.discount*np.dot(self.T,V_n)),axis = 0)
            iterId = iterId +1
            epsilon = np.linalg.norm(np.subtract(V_n,V_n1),np.inf)
            if epsilon < tolerance:
                break
            tempV = V_n
            V_n = V_n1
            V_n1 = tempV
            
        return [policyN,V_n,iterId,epsilon]
        