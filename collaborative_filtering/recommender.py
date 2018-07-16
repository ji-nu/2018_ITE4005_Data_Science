"""
Data Science (ITE4005)
Programming Assignment #4 : Predict ratings of movie

Author : Jinwoo Jang
Student No. : 2013021812
Email : real.longrain@gmail.com

Interpreter Version : Python 3.5.2
"""

import numpy as np
import sys


def parse(data_path):
    with open(data_path, 'r') as f:
        return np.array(list(map(lambda s: s.strip().split(), f.readlines()))).astype(int)


class Recommender(object):

    def __init__(self, train_data_path: str):
        self._data = parse(train_data_path)

        self.item_num = max(self._data[:, 1])
        self.user_num = max(self._data[:, 0])
        self.rating_max = np.max(self._data[:, 2])
        self.rating_min = np.min(self._data[:, 2])

        # (-1) means unrated item
        self.rating_mat = np.reshape(np.array([-1] * self.user_num * self.item_num),
                                     [self.user_num, self.item_num])

        for row in self._data:
            user, item, rating, _ = row
            self.rating_mat[user - 1][item - 1] = rating

        self.avg_rating = self._get_avg_rating_matrix()
        self.user_sim_mat = self._get_user_similarity_matrix(similarity=self._cosine)

    def _cosine(self, vec1, vec2):
        same_item_idx = (vec1 >= 0) * (vec2 >= 0)
        if len(same_item_idx[same_item_idx]) == 0:
            return 0
        svec1 = vec1[same_item_idx]
        svec2 = vec2[same_item_idx]
        return svec1.dot(svec2) / (sum(svec1 ** 2) + sum(svec2 ** 2))

    def _get_avg_rating_matrix(self):
        avg_rating = np.zeros(self.user_num)
        for user in range(self.user_num):
            rated_item = self.rating_mat[user] >= 0
            avg_rating[user] = sum(self.rating_mat[user][rated_item]) / len(rated_item[rated_item])
        return avg_rating

    def _get_user_similarity_matrix(self, similarity):
        sim_mat = np.zeros([self.user_num, self.user_num])
        for i in range(len(self.rating_mat)):
            for j in range(i + 1, len(self.rating_mat)):
                sim_mat[i][j] = sim_mat[j][i] = similarity(self.rating_mat[i], self.rating_mat[j])
        return sim_mat

    def aggregate_rating(self, item, user):
        try:
            item_info = self.rating_mat[:, item]
            rated = item_info >= 0
            rated_user_sim = self.user_sim_mat[user, rated]
            item_ratings = item_info[rated] - self.avg_rating[rated]

            # when user have no neighbor.
            if sum(rated_user_sim) == 0:
                return self.avg_rating[user]

            rating = self.avg_rating[user] + (sum(rated_user_sim * item_ratings) / sum(rated_user_sim))

            return np.clip(rating, self.rating_min, self.rating_max)

        except IndexError:  # when encounter new item
            return self.avg_rating[user]


if __name__=='__main__':

    BASE_DATA = sys.argv[1]
    TEST_DATA = sys.argv[2]

    print('Base data : %s' % BASE_DATA)
    print('Test data : %s' % TEST_DATA)
    print('Parsing data and pre-processing...')
    rc = Recommender(BASE_DATA)

    with open(BASE_DATA + '_prediction.txt', 'w') as out:
        test = parse(TEST_DATA)
        for i in range(len(test)):
            user, item, _, _ = test[i]
            r = rc.aggregate_rating(item - 1, user - 1)
            out.write('%d\t%d\t%f\n' % (user, item, r))
            sys.stdout.write('\rPredicting... %d%% (%d/%d)' % (((i+1) / len(test)) * 100, i + 1, len(test)))
            sys.stdout.flush()

    print('\nPredict output saved at %s' % (BASE_DATA + '_prediction.txt'))
