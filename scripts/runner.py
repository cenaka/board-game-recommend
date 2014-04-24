import csv
import datetime
from item_based_knn import ItemBasedKNNModel
from svd import SVDModel

__author__ = 'jambo'
import numpy as np

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def mae(predictions, targets):
    return np.abs((predictions - targets)).mean()

class DataSetMovieLens100K(object):
    """
    MovieLens100K Dataset loader
    """
    def __init__(self, train_set_path="resources/ml-100k/u1.base", test_set_path="resources/ml-100k/u1.test"):
        self.max_rating = 5
        self.n_users = 943
        self.n_items = 1682
        self.ratings = np.zeros((self.n_users, self.n_items))
        self.ratings_as_list = []
        self.tests = []
        self.answers = []
        for u, i, r, _ in csv.reader(file(train_set_path, "r"), dialect="excel-tab"):
            u, i, r = int(u) - 1, int(i) - 1, int(r)
            self.ratings[(u, i)] = r
            self.ratings_as_list.append((u, i, r))

        for u, i, r, _ in csv.reader(file(test_set_path, "r"), dialect="excel-tab"):
            u, i, r = int(u) - 1, int(i) - 1, int(r)
            self.tests.append((u, i))
            self.answers.append(r)
        self.answers = np.array(self.answers)

class DataSetMovieLens10M(object):
    def __init__(self, file_path, answers_file_path):
        with open(file_path, 'r') as file:
            self.max_rating, self.n_users, self.n_items, self.n_train_rates, self.n_test_rates \
                = map(int, file.readline().split())
            self.ratings = np.zeros((self.n_users, self.n_items))
            self.ratings_as_list = []
            self.tests = []
            for _ in xrange(self.n_train_rates):
                u, i, r = map(int, file.readline().split())
                self.ratings[(u, i)] = r
                self.ratings_as_list.append((u, i, r))
            for _ in xrange(self.n_test_rates):
                self.tests.append((map(int, file.readline().split())))  # user - item
            with open(answers_file_path, 'r') as answers_file:
                self.answers = np.array(map(int, answers_file.readlines()))




if __name__ == '__main__':

    # 10M Dataset loading
    # if len(sys.argv) > 1:
    #     dataset_number = sys.argv[1]
    # else:
    #     dataset_number = "1"
    # dataset = DataSet("movielensfold%s.txt" % dataset_number, "movielensfold%sans.txt" % dataset_number)

    dataset = DataSetMovieLens100K()
    print "dataset loaded"

    # SVD
    # model = SVDModel(dataset, 5, 0.05, 0.005)

    model = ItemBasedKNNModel(dataset)
    time = datetime.datetime.now()
    print "start fitting model..."
    model.load_similarity_matrix()
    model.fit_model()
    # model.drop_similarity_matrix()
    print "fitting model: seconds_passed: %s" % (datetime.datetime.now() - time).total_seconds()

    predicted = np.array([model.predict(u, i) for u, i in model.dataset.tests])
    print "rmse: %0.5f" % rmse(predicted, model.dataset.answers)
    print "mae: %0.5f" % mae(predicted, model.dataset.answers)

    # grid_search()
