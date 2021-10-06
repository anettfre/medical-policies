import numpy as np
import pandas

import thompson_bandit
import data_generation
np.random.seed(42)
def reward_function(action, outcome):
    return -0.1 * (action!=0) + outcome

def gene_exploration(generator, treatment, T, reward_function=reward_function, everyone=False):
    patients = []
    for t in range(T):
        x = generator.generate_features()
        a = treatment
        # == 1 and x[0][44] == 1 and x[0][24] == 1 and x[0][50] == 1 and x[0][23] == 1 and x[0][101] == 1:
        """
        if x[0][72] == 1:
            y = generator.generate_outcome(x, 1)
        else:
            y = generator.generate_outcome(x, a)
        """
        y = generator.generate_outcome(x, a)
        r = reward_function(a, y)
        if everyone:
            patients.append(x[0])
        else:
            if r > 0:
                patients.append(x[0])
    genes = []
    for patient in patients:
        genes.append(np.where(patient == 1))
    most_common_genes = {}
    for i in genes:
        for j in i[0]:
            if j in most_common_genes:
                most_common_genes[j] += 1
            else:
                most_common_genes[j] = 0
    most_common_genes = {key: value for key, value in sorted(most_common_genes.items(), key=lambda item: item[1], reverse=True)}
    return len(patients), most_common_genes

def test_policy_additional(generator, policy,  T, reward_function=reward_function):
    print("Additional treatments testing for ", T, "steps")
    policy.set_reward(reward_function)
    u = 0
    not_placebo = 0 #counting the people not given placebo
    number_of_treatments = 129
    actioncount = np.zeros(number_of_treatments)
    for t in range(T):
        x = generator.generate_features()
        a = policy.recommend(x)
        actioncount[a] += 1
        if (a > 0): 
            not_placebo += 1
        y = generator.generate_outcome(x, a)
        r = reward_function(a, y)
        u += r
        policy.observe(x, a, y)
        #print(a)
        #print("x: ", x, "a: ", a, "y:", y, "r:", r)
        
    return [u/T, not_placebo, actioncount]

if __name__ == "__main__":
    prior_a = 1
    prior_b = 1
    policy = thompson_bandit.ThompsonBandit(129, 2, prior_a, prior_b)
    generator = data_generation.DataGenerator(matrices="./big_generating_matrices.mat")
    """
    T = 100000
    result = test_policy_additional(generator, policy, T)
 
    b = np.argsort(-result[2])
    for first in b:
        print("Treatment ", first, "used in patients ", result[2][first], " number of times. ")
   

    print("Utility: %f" %result[0])
    """
    n_patients, most_common_genes = gene_exploration(generator, 2, 100000, everyone=False)

    print(n_patients)
    print(most_common_genes[72])


## Gene = 72, treatment 1. Gene = 55, treatment 1. Gene = 83
"""
features = pandas.read_csv('data/medical/historical_X.dat', header=None, sep=" ").values
actions = pandas.read_csv('data/medical/historical_A.dat', header=None, sep=" ").values
outcome = pandas.read_csv('data/medical/historical_Y.dat', header=None, sep=" ").values
observations = features[:, :128]
labels = features[:,128] + features[:,129]*2

import data_generation
import random_recommender
policy_factory = random_recommender.RandomRecommender
import reference_recommender
#policy_factory = reference_recommender.HistoricalRecommender

## First test with the same number of treatments
print("---- Testing with only two treatments ----")

print("Setting up simulator")
generator = data_generation.DataGenerator(matrices="./big_generating_matrices.mat")
print("Setting up policy")
policy = policy_factory(generator.get_n_actions(), generator.get_n_outcomes())
## Fit the policy on historical data first
print("Fitting historical data to the policy")
policy.fit_treatment_outcome(features, actions, outcome)
## Run an online test with a small number of actions
print("Running an online test")
n_tests = 1000
result = test_policy(generator, policy, default_reward_function, n_tests)
print("Total reward:", result)
print("Final analysis of results")
policy.final_analysis()

## First test with the same number of treatments
print("--- Testing with an additional experimental treatment and 126 gene silencing treatments ---")
print("Setting up simulator")
generator = data_generation.DataGenerator(matrices="./big_generating_matrices.mat")
print("Setting up policy")
policy = policy_factory(generator.get_n_actions(), generator.get_n_outcomes())
## Fit the policy on historical data first
print("Fitting historical data to the policy")
policy.fit_treatment_outcome(features, actions, outcome)
## Run an online test with a small number of actions
print("Running an online test")
n_tests = 1000
result = test_policy(generator, policy, default_reward_function, n_tests)
print("Total reward:", result)
print("Final analysis of results")
policy.final_analysis()
"""
