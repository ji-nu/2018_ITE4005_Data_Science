"""
Data Science (ITE4005)
Programming Assignment #3 : Clustering by DBSCAN

Author : Jinwoo Jang
Student No. : 2013021812
Email : real.longrain@gmail.com

Interpreter Version : Python 3.5.2
"""

# import matplotlib.pyplot as plt
import sys
from math import sqrt


class Point:
    def __init__(self, id, x: float, y: float):
        self.id = int(id)
        self.x = x
        self.y = y
        self.label = None


class DBSCAN:
    def __init__(self, data_sets: list, eps: float, min_pts: int):
        self.data_sets = data_sets
        self.eps = eps
        self.min_pts = min_pts

    @staticmethod
    def distance(pt_lhs: Point, pt_rhs: Point) -> float:
        """
        Get L2 distance between two points
        :param pt_lhs: point1
        :param pt_rhs: point2
        :return: distance
        """
        return sqrt((pt_lhs.x - pt_rhs.x) ** 2 + (pt_lhs.y - pt_rhs.y) ** 2)

    def get_neighbor(self, curr_pt: Point) -> list:
        """
        Get points in an eps-neighborhood of a given data
        :param curr_pt: candidate core point
        :return: list of neighbor points
        """
        neighbor = [p for p in self.data_sets if self.distance(curr_pt, p) <= self.eps]
        neighbor.remove(curr_pt)
        return neighbor

    def run(self):
        cluster_count = 0
        clusters = []
        for pt in self.data_sets:
            if pt.label is not None:  # already processed point
                continue

            neighbor = self.get_neighbor(pt)

            if len(neighbor) < self.min_pts:
                pt.label = 0  # Noise
                continue

            cluster_count += 1  # increase number of cluster
            clusters.append([])

            pt.label = cluster_count
            clusters[-1].append(pt)

            seed = set(neighbor)

            while seed:
                q = seed.pop()
                if q.label == 0:  # Noise
                    q.label = cluster_count
                if q.label is not None:
                    continue

                q.label = cluster_count
                clusters[-1].append(q)
                neighbor = self.get_neighbor(q)

                if len(neighbor) >= self.min_pts:
                    seed.update(neighbor)

            print('cluster %d, member = %d' % (cluster_count, len(clusters[-1])))

        return clusters


if __name__ == '__main__':

    INPUT_FILE_NAME = sys.argv[1]
    CLUSTER_NUMBER = int(sys.argv[2])
    EPS = float(sys.argv[3])
    MIN_PTS = int(sys.argv[4])

    # Read data
    data = []
    for d in open(INPUT_FILE_NAME, 'r').readlines():
        data.append(Point(*map(float, d.strip().split())))

    # Do DBSCAN
    dbscan = DBSCAN(data, EPS, MIN_PTS)
    print('start clustering')
    clusters = dbscan.run()
    clusters.sort(key=len)  # sort cluster by the number of it's object members
    if len(clusters) > CLUSTER_NUMBER:
        print('Found %d clusters, discard %d clusters' % (len(clusters), len(clusters) - CLUSTER_NUMBER))
        clusters = clusters[len(clusters) - CLUSTER_NUMBER:]

    count = 0
    for c in clusters:
        c.sort(key=lambda p: p.id)
        f = open('input%d_cluster_%d.txt' % (int(INPUT_FILE_NAME[-5]), count), 'w')
        for p in c:
            f.write(str(p.id) + '\n')
            # plt.scatter(p.x, p.y, s=1.5, c='C' + str(p.label))
        count += 1
        f.close()
    # plt.show()