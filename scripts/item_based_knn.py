"""
Item-based collaborative filtering
"""
__author__ = 'jambo'

import math
import scipy.stats
import numpy as np

def similarity(item1, item2):
    """
    Pierson Correlation coefficient
    """
    # r_ui, r_uj
    filtered_item1, filtered_item2 = item1[np.logical_and(item1 > 0, item2 > 0)], \
                                     item2[np.logical_and(item1 > 0, item2 > 0)]

    if np.alen(filtered_item1) == 0 or np.alen(filtered_item2) == 0:
        return 0

    item1_ratings = item1[item1 > 0]
    item2_ratings = item2[item2 > 0]
    if np.alen(item1_ratings) == 0 or np.alen(item2_ratings) == 0:
        return 0

    item1_mean = np.mean(item1_ratings)
    item2_mean = np.mean(item2_ratings)
    lower = np.sqrt(np.mean((filtered_item1 - item1_mean) ** 2)) * np.sqrt(np.mean((filtered_item2 - item2_mean) ** 2))
    upper = np.mean((filtered_item1 - item1_mean) * (filtered_item2 - item2_mean))
    return upper / lower if lower != 0 else 0
    # As alternative you can use library function
    # corr = scipy.stats.pearsonr(filtered_item1, filtered_item2)[0]
    # return corr if not math.isnan(corr) else 0


class ItemBasedKNNModel(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.similarity_matrix = None
        self.user_item_matrix = self.dataset.ratings
        self.item_user_matrix = self.user_item_matrix.T

    def fit_model(self):
        if self.similarity_matrix is None:
            self._init_similarity_matrix()
        self.means = []
        for i in xrange(self.dataset.n_items):
            i_ = self.item_user_matrix[i][self.item_user_matrix[i] > 0]
            self.means.append(np.mean(i_) if not np.alen(i_) == 0 else 0)

    def _init_similarity_matrix(self):
        self.similarity_matrix = np.zeros((self.dataset.n_items, self.dataset.n_items))
        for i in xrange(self.dataset.n_items):
            for j in xrange(self.dataset.n_items):
                self.similarity_matrix[(i, j)] = similarity(self.item_user_matrix[i], self.item_user_matrix[j])

    def drop_similarity_matrix(self, filename="resources/similarity.txt"):
        np.savetxt(filename, self.similarity_matrix)

    def load_similarity_matrix(self, filename="resources/similarity.txt"):
        self.similarity_matrix = np.loadtxt(filename)

    def predict(self, user_id, item_id):
        average_item_rating = self.means[item_id]
        i = item_id
        items = np.arange(0, self.dataset.n_items)
        neighbours = items[self.user_item_matrix[user_id] > 0]
        # neighbours = neighbours[np.argsort(abs(self.similarity_matrix[item_id, neighbours]))]
        upper_sum = np.sum([
            self.similarity_matrix[(i, j)] * (self.item_user_matrix[(j, user_id)] - self.means[j])
            for j in neighbours])
        lower_sum = np.sum(np.abs([self.similarity_matrix[(i, j)] for j in neighbours]))
        return average_item_rating if lower_sum == 0 else average_item_rating + upper_sum / lower_sum


    # item1 = item_user_matrix[item_id]
    # r_u_avg = means[item_id]
    # i = item_id
    # items = np.arange(0, ITEM_NUMBER)
    # neighbours = items[user_item_matrix[user_id] > 0]
    # neighbours = neighbours[np.argsort(abs(similarity_matrix[item_id, neighbours]))]
    #
    # # for j in neighbours:
    # #     print similarity_matrix[(i, j)] #* (item_user_matrix[(j, user_id)] - np.mean(item_user_matrix[j][item_user_matrix[j] > 0]))
    # upper_sum = np.sum([
    #     similarity_matrix[(i, j)] * (item_user_matrix[(j, user_id)] - means[j])
    #     for j in neighbours])
    # lower_sum = np.sum(np.abs([similarity_matrix[(i, j)] for j in neighbours]))
    # #print r_u_avg if lower_sum == 0 else r_u_avg + upper_sum / lower_sum
    # if lower_sum == 0:
    #     print "BAM"
    #     return r_u_avg
    # else:
    #     return r_u_avg + upper_sum / lower_sum

