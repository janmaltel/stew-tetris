import numpy as np
import stew.example_data as create
import stew.mlogit as mlogit
import stew.utils as utils

np.random.seed(5)

num_features = 3
num_choices = 10
max_cv_splits = 10
num_lambdas = 30
num_states = 1000

data = create.discrete_choice_example_data(num_states=num_states, num_features=num_features, num_choices=num_choices,
                                           probabilistic=True)
np.std(data, axis=0)
mlog = mlogit.StewMultinomialLogit(num_features=num_features, max_splits=max_cv_splits, num_lambdas=num_lambdas,
                                   nonnegative=True)
print(mlog.fit(data=data, lam=0, standardize=False))
print(mlog.fit(data=data, lam=0, standardize=True))

print(mlog.fit(data=data, lam=100, standardize=False))
print(mlog.fit(data=data, lam=100, standardize=True))
predicted_choices = mlog.predict(new_data=np.delete(data, 1, 1))
utils.multi_class_error(predicted_choices, data[:, 1])
# print(mlog.fit(data=data, start_weights=mlog.weights, lam=100000))
weights, cv_min_lambda = mlog.cv_fit(standardized_data=data)




