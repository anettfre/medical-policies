import math

class UCB1:
    def __init__(self, n_actions, n_outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        #initialize number of arms
        self.counts = [0 for col in range(n_actions)]
        self.values = [0.0 for col in range(n_actions)]
        self.reward = self._default_reward
        self.chosen_arm = 0 

    def _default_reward(self, action, outcome):
        return outcome

    def set_reward(self, reward):
        self.reward = reward

    def estimate_utility(self, data, actions, outcome, policy=None):
        return 0


    # UCB arm selection based on max of UCB reward of each arm
    def recommend(self, x):
        n_arms = len(self.counts)
        for arm in range(n_arms):
            if self.counts[arm] == 0:
                self.chosen_arm = arm
                return arm
    
        ucb_values = [0.0 for arm in range(n_arms)]
        total_counts = sum(self.counts)
        
        for arm in range(n_arms):
            bonus = math.sqrt((2 * math.log(total_counts)) / float(self.counts[arm]))
            ucb_values[arm] = self.values[arm] + bonus
        self.chosen_arm = ucb_values.index(max(ucb_values))
        return ucb_values.index(max(ucb_values))
        
    # Choose to update chosen arm and reward
    def observe(self, user=None, action=None, outcome=None):
        
        #self.counts[action] = self.counts[self.chosen_arm] + 1
        #n = self.counts[self.chosen_arm]
        
        # Update average/mean value/reward for chosen arm
        #value = self.values[self.chosen_arm]
        #new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * self.reward(action, outcome)
        #self.values[self.chosen_arm] = new_value
        return


class linucb():
    
    def __init__(self, K_arms, d, alpha=0.1):
        self.K_arms = K_arms
        self.linucb_arms = [linucb_disjoint_arm(arm_index = 1, d = d, alpha = alpha) for i in range(K_arms)]
    
    def set_reward(self, reward):
        self.reward = reward 

    def select_arm(self, x_array):
        # Initiate ucb to be 0
        highest_ucb = -1
        
        # Track index of arms to be selected on if they have the max UCB.
        candidate_arms = []
        
        for arm_index in range(self.K_arms):
            # Calculate ucb based on each arm using current covariates at time t
            arm_ucb = self.linucb_arms[arm_index].calc_UCB(x_array)
            
            # If current arm is highest than current highest_ucb
            if arm_ucb > highest_ucb:
                
                # Set new max ucb
                highest_ucb = arm_ucb
                
                # Reset candidate_arms list with new entry based on current arm
                candidate_arms = [arm_index]

            # If there is a tie, append to candidate_arms
            if arm_ucb == highest_ucb:
                
                candidate_arms.append(arm_index)
        
        # Choose based on candidate_arms randomly (tie breaker)
        chosen_arm = np.random.choice(candidate_arms)
        
        return chosen_arm

    def observe(self, user, action, outcome):
        # Reshape covariates input into (d x 1) shape vector
        x = x_array.reshape([-1,1])
        
        # Update A which is (d * d) matrix.
        self.A += np.dot(x, x.T)
        
        # Update b which is (d x 1) vector
        # reward is scalar
        self.b += reward * x