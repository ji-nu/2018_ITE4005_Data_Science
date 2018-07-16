"""
Data Science (ITE4005)
Programming Assignment #2 : Decision Tree

Author : Jinwoo Jang
Student No. : 2013021812
Email : real.longrain@gmail.com

Interpreter Version : Python 3.5.2
"""

from math import log2, inf, sqrt

import random
import sys


class Node:
    """
    Decision Tree Node
    """
    def __init__(self, data_set: list):
        self.data_set = data_set
        # only leaf nodes can have own data set
        self.attr_idx = None
        # index of attribute split current node
        self.children = None
        # dict of children nodes
        self.label = None
        # major label of this node
        self.pos = None
        self.neg = None


class DecisionTree:
    def __init__(self, attribute_names: tuple, training_set: list, pessimistic_factor: float):
        """
        :param attribute_names: tuple of attribute names
        :param training_set: list of training data tuples
        :param pessimistic_factor : hyper parameter of model
        """
        self._attribute_names = attribute_names
        self._training_set = training_set
        self._pessimistic_factor = pessimistic_factor
        self._build_tree()
        self._post_pruning()

    @staticmethod
    def _entropy(data_set: list) -> float:
        """
        Computes the expected information (Info (D)) of the data set.
        :param data_set: A tuple set to measure the expected information
        :return: entropy (expected information)
        """
        label_prob = map(lambda c: c / len(data_set), DecisionTree._label_count(data_set).values())
        return -sum(map(lambda p: p * log2(p), label_prob))

    @staticmethod
    def _label_count(data_set: list) -> dict:
        """
        Count the number of tuples belonging to each label
        :param data_set: A tuple set to count
        :return: dict {label name : number of tuple}
        """
        label_column = [t[-1] for t in data_set]
        label_set = sorted(list(set(label_column)))
        label_count = {label: label_column.count(label) for label in label_set}
        return label_count

    @staticmethod
    def _major_pos_neg(data_set: list) -> (str, int, int):
        """
        Get major label name, the number of tuples belonging (pos) and others (neg)
        :param data_set: A tuple set to count
        :return: major label name, pos, neg
        """
        d = DecisionTree._label_count(data_set)
        major_label = max(d.keys(), key=(lambda key: d[key]))
        pos = d[major_label]
        neg = len(data_set) - pos
        return major_label, pos, neg

    @staticmethod
    def _entropy_a(split_set: list) -> float:
        """
        Computes the expected information (Info_A (D)) of the split data set.
        :param split_set: data set list
        :return: entropy (expected information) of split data set
        """
        overall_size = sum(map(len, split_set))
        entropy_list = [DecisionTree._entropy(data_set) * len(data_set) for data_set in split_set]
        return sum(map(lambda e: e / overall_size, entropy_list))

    @staticmethod
    def _gain(data_set: list, split_set: list) -> float:
        """
        Calculate the information gain due to the branch by specific attribute
        :param data_set: data set to split
        :param split_set: split data set
        :return: information gain (entropy change)
        """
        return DecisionTree._entropy(data_set) - DecisionTree._entropy_a(split_set)

    @staticmethod
    def _split_data_set(data_set: list, attr_idx: int) -> dict:
        """
        Create a dictionary with data sets that split by specific attributes.
        :param data_set: data set to split
        :param attr_idx: index of attribute for split
        :return: dictionary with split data sets
        """
        categories = set([t[attr_idx] for t in data_set])
        split = {c: [] for c in categories}
        list(map(lambda t: split[t[attr_idx]].append(t), data_set))
        return split

    @staticmethod
    def _split_info(data_set: list, split_set: list) -> float:
        """
        Calculate split information
        :param data_set: data set to split
        :param split_set: split data set
        :return: split information
        """
        dl = len(data_set)
        return -sum(map(lambda s: (len(s) / dl) * log2(len(s) / dl), split_set))

    @staticmethod
    def _gain_ratio(data_set: list, attr_idx: int) -> float:
        """
        Calculate gain ratio
        :param data_set: data set to split
        :param attr_idx: index of attribute for split
        :return: gain ratio
        """
        split_set = list(DecisionTree._split_data_set(data_set, attr_idx).values())

        if len(split_set) == 1:
            return -inf

        return DecisionTree._gain(data_set, split_set) / DecisionTree._split_info(data_set, split_set)

    def _split_node(self, node: Node, attr_idxs: list):
        """
        Split current node by a best split attribute
        :param node: node to split
        :param attr_idxs: information of available attributes
        :return:
        """
        # stop split when reach homogeneous node
        if DecisionTree._entropy(node.data_set) == 0:
            node.label, node.pos, node.neg = DecisionTree._major_pos_neg(node.data_set)
            return

        # stop split when current amount of tuple less than specific number
        node.label, node.pos, node.neg = DecisionTree._major_pos_neg(node.data_set)
        if node.pos + node.neg < len(self._training_set) * 0.004:  # pre-pruning factor
            return

        # find best split attribute
        max_gain_ratio = -inf

        split_idx = None
        for idx in attr_idxs:
            gain = DecisionTree._gain_ratio(node.data_set, idx)
            if gain > max_gain_ratio:
                max_gain_ratio = gain
                split_idx = idx

        # split current node by best attribute
        split_dict = DecisionTree._split_data_set(node.data_set, split_idx)
        node.children = {category: Node(split_dict[category]) for category in split_dict.keys()}
        node.attr_idx = int(split_idx)
        node.data_set = None
        attr_idxs.remove(split_idx)

        # split node recursively for children
        for child in node.children.values():
            self._split_node(child, list(attr_idxs))

    def _build_tree(self):
        """
        Build decision tree
        :return:
        """
        self.root = Node(self._training_set)
        self._split_node(self.root, list(range(len(self._attribute_names) - 1)))

    @staticmethod
    def _error_estimate(node: Node):
        """
        Get error based on C4.5's method
        :param node: node to calculate error
        :return: error
        """
        n = node.neg + node.pos
        f = node.neg / n
        z = 0.69
        error = (f + ((z * z) / (2 * n)) +
                 z * sqrt((f / n) - ((f * f) / n) + ((z * z) / (4 * n * n)))) / (1 + ((z * z) / n))
        return error

    def _need_pruning(self, node: Node):
        """
        Check current node need to prune children nodes
        :param node: node to decide whether to pruning
        :return: True or False
        """
        if not node.children:  # leaf node
            return False

        n = node.neg + node.pos
        parent_error = DecisionTree._error_estimate(node)

        children_error = 0
        for child in node.children.values():
            child_error = ((child.neg + child.pos) / n) * DecisionTree._error_estimate(child)
            children_error += child_error

        return children_error > parent_error * self._pessimistic_factor

    def _post_pruning(self):
        """
        Prune the fully grown tree
        """
        self._pruning_subtree(self.root)

    def _pruning_subtree(self, node: Node, depth=0):
        """
        Prune the tree recursively (bottom-up)
        """
        # prune child first
        if node.children:
            for child in node.children.values():
                self._pruning_subtree(child, depth + 1)

        # prune current node
        if self._need_pruning(node):
            node.children = None

    def classification(self, test: list):
        """
        Do classification on a tuple
        :param test: a tuple to do classification
        :return:
        """
        node = self.root
        while node.children:
            category = test[node.attr_idx]
            tmp = node
            try:
                node = node.children[category]
            except KeyError:  # when there no matching attribute
                node = tmp
                break
        return node.label

    def test(self, test_set: list, answer_set: list):
        """
        test accuracy of model for test data
        :param test_set: test data set
        :param answer_set: answer data set
        :return: accuracy of model
        """
        n, correct = 0, 0
        for td, ad in test_set, answer_set:
            n += 1
            if self.classification(td) == ad[-1]:
                correct += 1
        return correct / n

    def validate(self, validation_set: list):
        """
        test accuracy of model for validation data
        :param validation_set: validation data set
        :return: accuracy of model
        """
        n, correct = 0, 0
        for vd in validation_set:
            n += 1
            if self.classification(vd[:-1]) == vd[-1]:
                correct += 1
        return correct / n


class Tuner:
    """
    Do Cross-Validation to tuning hyper-parameter
    """
    def __init__(self, attr_name: tuple, train_set: list, k: int=10):
        self.train_set = train_set
        self.k = k
        self.attr_name = attr_name
        self.fine_pessimistic = 1.00
        random.shuffle(train_set)

    def _k_fold(self):
        """
        Split training data into k training - validation data sets
        :return: list of folded training, validation data set (map object)
        """
        n = int(len(self.train_set) / self.k)
        ts = self.train_set
        return map(lambda i: (ts[:i * n] + ts[(i + 1) * n:], ts[i * n: (i + 1) * n]), range(self.k))

    def get_model(self):
        """
        Get a tuned decision tree model
        :return: DecisionTree
        """
        pessimistic = 1.00
        max_accuracy = -inf
        tuning_epoch = 10
        step_size = 0.01

        for i in range(self.k):
            print('Tuning model #%d of %d ...' % (i + 1, tuning_epoch))
            accuracy = 0

            # get average accuracy of current pessimistic parameter
            for train, validation in self._k_fold():
                model = DecisionTree(self.attr_name, train, pessimistic)
                a = model.validate(validation)
                accuracy += a

            accuracy = accuracy / self.k
            print(accuracy)

            if accuracy > max_accuracy * 1.0001:
                max_accuracy = accuracy
                self.fine_pessimistic = pessimistic
            pessimistic += step_size

        return DecisionTree(self.attr_name, self.train_set, self.fine_pessimistic)


if __name__ == '__main__':

    def data_parser(filename: str):
        with open(filename, 'r') as ts:
            data = list(map(str.split, ts.readlines()))
            return tuple(data[0]), data[1:]

    TRAIN_SET_DATA = sys.argv[1]
    TEST_SET_DATA = sys.argv[2]
    RESULT_DATA = sys.argv[3]

    print('Training data : %s' % TRAIN_SET_DATA)
    print('Test data : %s' % TEST_SET_DATA)
    print('Result data : %s' % RESULT_DATA)

    # Read training data set
    attr_name, train_set = data_parser(TRAIN_SET_DATA)
    # Get decision tree
    model = Tuner(attr_name, train_set).get_model()
    # Read test data set
    _, test_set = data_parser(TEST_SET_DATA)

    # Write classification result
    out = open(RESULT_DATA, 'w')
    out.write('\t'.join(attr_name) + '\n')
    for td in test_set:
        td.append(model.classification(td))
        s = '\t'.join(map(str, td)) + '\n'
        out.write(s)
    out.close()
