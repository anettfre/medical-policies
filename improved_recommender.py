# -*- Mode: python -*-
# A simple reference recommender
#
#
# This is a medical scenario with historical data. 
#
# General functions
#
# - set_reward
# 
# There is a set of functions for dealing with historical data:
#
# - fit_data
# - fit_treatment_outcome
# - estimate_utiltiy
#
# There is a set of functions for online decision making
#
# - predict_proba
# - recommend
# - observe

from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
np.random.seed(42)
class ImprovedRecommender:

    #################################
    # Initialise
    #
    # Set the recommender with a default number of actions and outcomes.  This is
    # because the number of actions in historical data can be
    # different from the ones that you can take with your policy.
    def __init__(self, n_actions, n_outcomes):
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward
        self.scaler = StandardScaler()


    ## By default, the reward is just equal to the outcome, as the actions play no role.
    def _default_reward(self, action, outcome):
        return outcome

    # Set the reward function r(a, y)
    def set_reward(self, reward):
        self.reward = reward
    
    ##################################
    # Fit a model from patient data.
    #
    # This will generally speaking be an
    # unsupervised model. Anything from a Gaussian mixture model to a
    # neural network is a valid choice.  However, you can give special
    # meaning to different parts of the data, and use a supervised
    # model instead.
    def fit_data(self, data):
        print("Preprocessing data")
        return None


    ## Fit a model from patient data, actions and their effects
    ## Here we assume that the outcome is a direct function of data and actions
    ## This model can then be used in estimate_utility(), predict_proba() and recommend()
    def fit_treatment_outcome(self, data, actions, outcome):
        if len(data) > 1:
            #First iteration
            self.scaler.fit(data)
            self.data = self.scaler.transform(data)
            self.actions = actions
            self.outcome = outcome
        else:
            self.data = np.concatenate((self.data, data), axis=0)
            self.actions = np.concatenate((self.actions, np.array([actions]).reshape(1,1)), axis=0)
            self.outcome = np.concatenate((self.outcome, np.array([outcome]).reshape(1,1)), axis=0)
        #Scaling the data to get normalized features
        self.model = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5,2), random_state=0, max_iter=2000)
        fit_data = pd.DataFrame(self.data)
        fit_data['a'] = self.actions
        self.model.fit(fit_data.values, np.ravel(self.outcome))
        return None

    ## Estimate the utility of a specific policy from historical data (data, actions, outcome),
    ## where utility is the expected reward of the policy.
    ##
    ## If policy is not given, simply use the average reward of the observed actions and outcomes.
    ##
    ## If a policy is given, then you can either use importance
    ## sampling, or use the model you have fitted from historical data
    ## to get an estimate of the utility.
    ##
    ## The policy should be a recommender that implements get_action_probability()
    def estimate_utility(self, data, actions, outcome, policy=None):
        if policy == None:
            return sum(self.reward(actions, outcome))
        else:
            E_utility = 0
            print("Estimating")
            ## Iterating through data:
            for i in range(data.shape[0]):
                curr_data = np.array(data[i].reshape(1,130))

                recommended_action = policy.recommend(curr_data)
                action_proba = policy.predict_proba(curr_data, recommended_action)

                # E[f(X)] = \sum_x p(x)*f(x):
                E_reward = action_proba[0,0]*self.reward(recommended_action, 0) + action_proba[0,1]*self.reward(recommended_action, 1)

                E_utility += E_reward

            return E_utility
           

    # Return a distribution of effects for a given person's data and a specific treatment.
    # This should be an numpy.array of length self.n_outcomes
    def predict_proba(self, data, treatment):
        df = pd.DataFrame(data)
        df['a'] = treatment
        return self.model.predict_proba(df)

    # Return a distribution of recommendations for a specific user datum
    # This should a numpy array of size equal to self.n_actions, summing up to 1
    def get_action_probabilities(self, user_data):
        #print("Recommending")
        return np.ones(self.n_actions) / self.n_actions

    
    # Return recommendations for a specific user datum
    # This should be an integer in range(self.n_actions)
    def recommend(self, user_data):
        # Finding the probabilities of outcomes given user_data and action
        #print("recommending")
        scaled_user_data = self.scaler.transform(user_data)
        P_outcomes_placebo = self.predict_proba(scaled_user_data, 0)
        P_outcomes_drug = self.predict_proba(scaled_user_data, 1)

        # Estimating reward
        E_reward_placebo = P_outcomes_placebo[0,0]*self.reward(0, 0) + P_outcomes_placebo[0,1]*self.reward(0, 1)
        E_reward_drug = P_outcomes_drug[0,0]*self.reward(1, 0)+ P_outcomes_drug[0,1]*self.reward(1, 1)


        if (E_reward_placebo >= E_reward_drug):
            return 0
        else:
            return 1
    # Observe the effect of an action. This is an opportunity for you
    # to refit your models, to take the new information into account.
    def observe(self, user, action, outcome):
        #data = user.reshape(1,130)
        #data = self.scaler.transform(data)
        #self.fit_treatment_outcome(data, action, outcome)
        return None


    # After all the data has been obtained, do a final analysis. This can consist of a number of things:
    # 1. Recommending a specific fixed treatment policy
    # 2. Suggesting looking at specific genes more closely
    # 3. Showing whether or not the new treatment might be better than the old, and by how much.
    # 4. Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment
    def final_analysis(self):
        return None
