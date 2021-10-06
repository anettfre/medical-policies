import numpy as np
import pandas
np.random.seed(42)
def default_reward_function(action, outcome):
    return -0.1*(action!=0) + outcome

def test_policy(generator, policy, reward_function, T):
    print("Testing for ", T, "steps")
    policy.set_reward(reward_function)
    u = 0

    for t in range(T):
        x = generator.generate_features()
        a = policy.recommend(x)
        y = generator.generate_outcome(x, a)
        r = reward_function(a, y)
        u += r
        policy.observe(x, a, y)
        #print(a)
        #print("x: ", x, "a: ", a, "y:", y, "r:", r)
    return u/T

def final_analysis(policies, features, actions, outcome, generator, big_generator, n_tests):
    """
    Policy 0-2: Utility from historical data
                LogisticRegressionRecommender on historical data
                MlpRecommender on historical data
    Policy 3-7: HistoricalRecommender on new patients
                ImprovedRecommender on new patients
                ImprovedRecommenderBig on new patients with big matrices
                AdaptiveRecommender on new patients
                AdaptiveRecommenderBig on new patients with big matrices
    """
    utilities = []
    for policy in range(len(policies)):
        if policy == 0:
            #Utility from historical data
            #policies[policy].set_reward(default_reward_function)
            utilities.append(policies[policy].estimate_utility(features, actions, outcome))
            print("0. Utility for historical data: ", utilities[policy])
        if policy == 1:
            #Utility for LR Recommender
            policies[policy].fit_treatment_outcome(features, actions, outcome)
            policies[policy].set_reward(default_reward_function)
            utilities.append(policies[policy].estimate_utility(features, None, None, policies[policy]) / features.shape[0])
            print("1. Utility for LogisticRegressionRecommender on historical data: ", utilities[policy])
        if policy == 2:
            #Utility for MlpRecommender
            policies[policy].fit_treatment_outcome(features, actions, outcome)
            policies[policy].set_reward(default_reward_function)
            utilities.append(policies[policy].estimate_utility(features, None, None, policies[policy]) / features.shape[0])
            print("2. Utility for MlpRecommender on historical data: ", utilities[policy])
        if policy == 3:
            #Utility for HistoricalRecommender on new patients
            policies[policy].fit_treatment_outcome(features, actions, outcome)
            utilities.append(test_policy(generator, policies[policy], default_reward_function, n_tests))
            print("3. Utility for HistoricalRecommender on new patients: ", utilities[policy])
        if policy == 4:
            #Utility for ImprovedRecommender on new patients
            policies[policy].fit_treatment_outcome(features, actions, outcome)
            utilities.append(test_policy(generator, policies[policy], default_reward_function, n_tests))
            print("4. Utility for ImprovedRecommender on new patients: ", utilities[policy])
        if policy == 5:
            #Utility for ImprovedRecommenderBig on new patients
            policies[policy].fit_treatment_outcome(features, actions, outcome)
            utilities.append(test_policy(big_generator, policies[policy], default_reward_function, n_tests))
            print("5. Utility for ImprovedRecommender on new patients, with additional treatments: ", utilities[policy])
        if policy == 6:
            #Utility for AdaptiveRecommender on new patients
            policies[policy].fit_treatment_outcome(features, actions, outcome)
            utilities.append(test_policy(generator, policies[policy], default_reward_function, n_tests))
            print("6. Utility for AdaptiveRecommender on new patients: ", utilities[policy])
        if policy == 7:
            #Utility for AdaptiveRecommenderBig on new patients
            policies[policy].fit_treatment_outcome(features, actions, outcome)
            utilities.append(test_policy(big_generator, policies[policy], default_reward_function, n_tests))
            print("7. Utility for AdaptiveRecommender on new patients, with additional treatments: ", utilities[policy])
    print("The best policy appears to be number ", utilities.index(max(utilities)))
    print("It yields ", max(utilities) - utilities[0], "more expected utility than the historical policy.")
    print("This is an increase of ", (max(utilities)/utilities[0])*100, "percent!")
    return None

features = pandas.read_csv('../../data/medical/historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv('../../data/medical/historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv('../../data/medical/historical_Y.dat', header=None, sep=" ").values

observations = features[:, :128]
labels = features[:,128] + features[:,129]*2

import data_generation
# Importing Recommenders
import historical_recommender
import lr_recommender
import mlp_recommender
import improved_recommender
import adaptive_recommender
import improved_recommender_big
import adaptive_recommender_big

historical_factory = historical_recommender.HistoricalRecommender
lr_factory = lr_recommender.LogisticRegressionRecommender
mlp_factory = mlp_recommender.MlpRecommender
improved_factory = improved_recommender.ImprovedRecommender
adaptive_factory = adaptive_recommender.AdaptiveRecommender
improved_big_factory = improved_recommender_big.ImprovedRecommenderBig
adaptive_big_factory = adaptive_recommender_big.AdaptiveRecommenderBig

## First test with the same number of treatments
print("---- Testing with only two treatments ----")

print("Setting up simulator")
generator = data_generation.DataGenerator()
big_generator = data_generation.DataGenerator(matrices="./big_generating_matrices.mat")
#generator = data_generation.DataGenerator()
print("Setting up policies")
policies = []
policies.append(historical_factory(len(actions), len(outcome)))
policies.append(lr_factory(2, 2))
policies.append(mlp_factory(2, 2))
policies.append(historical_factory(generator.get_n_actions(), generator.get_n_outcomes()))
policies.append(improved_factory(generator.get_n_actions(), generator.get_n_outcomes()))
policies.append(improved_big_factory(big_generator.get_n_actions(), big_generator.get_n_outcomes()))
policies.append(adaptive_factory(generator.get_n_actions(), generator.get_n_outcomes()))
policies.append(adaptive_big_factory(big_generator.get_n_actions(), big_generator.get_n_outcomes()))


n_tests = 1000

final_analysis(policies, features, actions, outcome, generator, big_generator, n_tests)

## Policies on historical data
"""
#Set rewards
lr_policy.set_reward(default_reward_function)
mlp_policy.set_reward(default_reward_function)

#policy = policy_factory(generator.get_n_actions(), generator.get_n_outcomes())
## Fit the policy on historical data first
print("Fitting historical data to the policy")
lr_policy.fit_treatment_outcome(features, actions, outcome)
mlp_policy.fit_treatment_outcome(features, actions, outcome)


## Run an online test with a small number of actions

print("Utility of historical data: ", hist_utility)

n_data = 1000
samples = len(outcome)
utilities = np.zeros(n_data)

for i in range(n_data):
    test = np.random.choice(samples, samples)
    test_outcome = outcome[test]
    test_action = actions[test]
    utilities[i] = historical_policy.estimate_utility(data=features, actions=test_action, outcome=test_outcome)

print("95 percent confidence interval for historical data: ", np.percentile(utilities, [2.5, 97.5]))
print("Calculating utility for improved policies")

print("MLP Classifier utility: ", mlp_utility)
print("Logistic Regression utility: ", lr_utility)
"""

## Confidence Intervals
"""
## 95% Confidence Interval with bootstrapping (Improved Policy)
utilities = np.zeros(n_tests)
for i in range(n_tests):
    print("Starting test", i)
    policy = mlp_factory(generator.get_n_actions(), generator.get_n_outcomes())
    test = np.random.choice(len(outcome), len(outcome))
    test_outcome = outcome[test]
    test_action = actions[test]
    policy.fit_treatment_outcome(features, test_action, test_outcome)
    utilities[i] = test_policy(generator, policy, default_reward_function, n_tests)
print(np.percentile(utilities, [2.5, 97.5]))

## 95% Confidence Interval with bootstrapping (Historical Policy)
utilities = np.zeros(n_tests)
for i in range(n_tests):
    print("Starting test", i)
    policy = historical_factory(generator.get_n_actions(), generator.get_n_outcomes())
    test = np.random.choice(len(outcome), len(outcome))
    test_outcome = outcome[test]
    test_action = actions[test]
    policy.fit_treatment_outcome(features, test_action, test_outcome)
    utilities[i] = test_policy(generator, policy, default_reward_function, n_tests)
print(np.percentile(utilities, [2.5, 97.5]))
"""
