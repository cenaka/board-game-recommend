import csv
import random
import datetime
import sys

__author__ = 'jambo'
import numpy as np

MAX_ITER = 100

class SVDModel:
    def __init__(self, dataset, num_of_factors, regularization_constant, learning_rate):
        self.dataset = dataset
        self.average = self._average_rating()
        self.b_users = np.zeros(dataset.n_users)
        self.b_items = np.zeros(dataset.n_items)
        self.p = np.random.random((dataset.n_users, num_of_factors)) - 0.5
        self.q = np.random.random((dataset.n_items, num_of_factors)) - 0.5
        self.regularization_constant = regularization_constant
        self.learning_rate = learning_rate
        self.validate_set_size = int(len(self.dataset.tests) * 0.2)
        self.size = len(self.dataset.tests)

    def predict(self, u, i):
        return self.average + self.b_users[u] + self.b_items[i] + np.inner(self.p[u], self.q[i])

    def fit_model(self):
        self._sgd()

    def rmse(self, cut=None):
        if cut is None:
            cut = self.size
        estimate = np.array([self.predict(u, i) for u, i in self.dataset.tests])[:cut]
        answers = self.dataset.answers[:cut]
        return float(np.sqrt(np.mean((estimate - answers) ** 2)))

    def _average_rating(self):
        return np.average(self.dataset.ratings[self.dataset.ratings > 0])

    def _error(self, u, i):
        return self.dataset.ratings[(u, i)] - self.predict(u, i)

    def validated_rmse(self):
        return self.rmse(cut=self.validate_set_size)

    def _sgd(self):
        gamma = self.learning_rate
        lam = self.regularization_constant
        previous_rmse = None
        for _ in xrange(MAX_ITER):
            random.shuffle(self.dataset.ratings_as_list)
            for u, i, r in self.dataset.ratings_as_list:
                error = self._error(u, i)
                new_b_u = self.b_users[u] + gamma * (error - lam * self.b_users[u])
                new_b_i = self.b_items[i] + gamma * (error - lam * self.b_items[i])
                new_p_u = self.p[u] + gamma * (error * self.q[i] - lam * self.p[u])
                new_q_i = self.q[i] + gamma * (error * self.p[u] - lam * self.q[i])
                self.b_users[u], self.b_items[i], self.p[u], self.q[i] = new_b_u, new_b_i, new_p_u, new_q_i
            new_rmse = self.validated_rmse()
            print "validate rmse: %0.5f" % new_rmse
            if previous_rmse is not None and previous_rmse - new_rmse < 5e-4:
                break

            previous_rmse = new_rmse


def grid_search(dataset):
    """
    Best Parameters searching
    """
    global results, learning_rate, factor_number, regularization_constant, model, time, rmse
    results = []
    for learning_rate in [0.005]:
        for factor_number in [0, 5, 10, 50, 100]:
            print "factor number = %d" % factor_number
            for regularization_constant in [0.05, 0.1, 0.5, 1, 5]:
                model = SVDModel(dataset, 50, regularization_constant, learning_rate)
                time = datetime.datetime.now()
                model.fit_model()
                print "seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds()
                rmse = model.rmse()
                print ("rmse for learning rate %0.4f and regularisation constant %0.4f: %0.5f"
                       % (learning_rate, regularization_constant, rmse))
                results.append((rmse, factor_number, learning_rate, regularization_constant))
    print "done"
    for rmse, factor_number, learning_rate, regularization_constant in sorted(results):
        print ("rmse for factor_number %d, learning rate %0.4f and regularisation constant %0.4f: %0.5f"
               % (factor_number, learning_rate, regularization_constant, rmse))












