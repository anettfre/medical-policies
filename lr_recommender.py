from sklearn import linear_model
from sklearn import preprocessing
import numpy as np
import pandas as pd

class LogisticRegressionRecommender:

    def __init__(self, n_actions, n_outcomes):
        """
        Set the recommender with a default number of actions and outcomes.
        """
        self.n_actions = n_actions
        self.n_outcomes = n_outcomes
        self.reward = self._default_reward

    def _default_reward(self, action, outcome):
        """
        By default, the reward is just equal to the outcome, as the actions play no role.
        """
        return outcome

    def set_reward(self, reward):
        """
        Set the reward function (r_t(a, y) = -.01a_t + y_t)
        """
        self.reward = reward

    def fit_data(self, data):
        """
        Fit a model from patient data.

        Using an unsupervised model: Gaussian mixture???

        (Could use a supervised model instead, to give special
        meaning to different parts of the data)
        """
        print("Preprocessing data")
        #self.model = linear_model.LogisticRegression()
        #self.model.fit(data)
        return None


    def fit_treatment_outcome(self, data, actions, outcome):
        """
        Fitting a LogisticRegression model from patient data, actions and their effects.

        Assuming that the outcome is a direct function of data and actions.

        (Can be used in estimate_utility(), predict_proba() and recommend() later)

        """
        print("Fitting treatment outcomes")

        # Making a Logistic Regression model
        self.model = linear_model.LogisticRegression(random_state=0, solver='liblinear')

        # Scaling data and combining data and actions in a dataframe
        scaled_data = preprocessing.scale(data)
        df = pd.DataFrame(scaled_data)
        df['a'] = actions

        # Training the model with action and outcome
        self.model.fit(df.values, np.ravel(outcome))

        return None


    def estimate_utility(self, data, actions, outcome, policy=None):
        """
         Estimating utility of a specific policy from historical data, where
         utility is the expected reward of policy.

         (The policy should be a recommender that implements get_action_probability())
        """
        
        if policy == None:
            #return average reward of observed actions and outcomes
            return sum(self.reward(actions, outcome))

        else:
            E_utility = 0

            ## Iterating through data:
            for i in range(data.shape[0]):
                curr_data = np.array(data[i].reshape(1,130))

                recommended_action = policy.recommend(curr_data)
                action_proba = policy.predict_proba(curr_data, recommended_action)

                # E[f(X)] = \sum_x p(x)*f(x):
                E_reward = action_proba[0,0]*self.reward(recommended_action, 0) + action_proba[0,1]*self.reward(recommended_action, 1)

                E_utility += E_reward

            return E_utility


    def predict_proba(self, data, treatment):
        """
        Return a distribution of effects for a given person's data and a specific treatment
        (should be a numpy.array with length self.n_outcomes)
        """
        df = pd.DataFrame(data)
        df['a'] = treatment
        return self.model.predict_proba(df)


    def get_action_probabilities(self, user_data):
        """
        Return a distribution of recommendations for a specific user datum

        (should a numpy.array of size equal to self.n_actions, summing up to 1)
        """
        #print("Recommending")
        return np.ones(self.n_actions) / self.n_actions



    def recommend(self, user_data):
        """
        Return recommendations for a specific user datum

        Choosing the action which yields the best estimated reward to maximize utility.

        (should be an integer in range(self.n_actions))
        """
        # Finding the probabilities of outcomes given user_data and action
        P_outcomes_placebo = self.predict_proba(user_data, 0)
        P_outcomes_drug = self.predict_proba(user_data, 1)

        # Estimating reward
        E_reward_placebo = P_outcomes_placebo[0,0]*self.reward(0, 0) + P_outcomes_placebo[0,1]*self.reward(0, 1)
        E_reward_drug = P_outcomes_drug[0,0]*self.reward(1, 0)+ P_outcomes_drug[0,1]*self.reward(1, 1)

        # Return the best action
        if (E_reward_placebo >= E_reward_drug):
            return 0
        else:
            return 1

        #return np.random.choice(self.n_actions, p = self.get_action_probabilities(user_data))


    def observe(self, user, action, outcome):
        """
        Observe the effect of an action.

        An opportunity for to refit our models, to take the new information into account.
        """
        #self.fit_treatment_outcome(user, action, outcome)
        return None


    def final_analysis(self):
        """
        1. Recommending a specific fixed treatment policy
        2. Suggesting looking at specific genes more closely
        3. Showing whether or not the new treatment might be better than the old, and by how much.
        4. Outputting an estimate of the advantage of gene-targeting treatments versus the best fixed treatment
        """
        return None
