#!/usr/bin/python3

"""
Data Science (ITE4005) Programming Assignment #1
Author : Jinwoo Jang
Student No. : 2013021812
Email : real.longrain@gmail.com

Interpreter Version : Python 3.5.2
"""

import sys
from itertools import chain, combinations
from math import floor

class Apriori:
    def __init__(self, min_sup, db_path, out_path):
        self.min_sup = min_sup / 100
        self.db_path = db_path
        self.out_path = out_path
        self.tr_size = 0

    def _transaction_generator(self):
        with open(self.db_path, 'r') as db:
            for transaction in db:
                items = map(int, transaction.split())
                yield list(items)

    def _get_frequent_information_size_1(self):
        candidate = dict()
        tr_counter = 0
        for tr in self._transaction_generator():
            tr_counter += 1
            for item in tr:
                if item not in candidate:
                    candidate[item] = 1
                else:
                    candidate[item] += 1
        self.tr_size = tr_counter
        # candidate key list 에 대하여, support > min_support 이상인 key 만 frequent list 에 저장
        frequents = list(filter(lambda i: candidate[i] / self.tr_size >= self.min_sup, candidate))
        # 저장한 frequent item 에 대한 support 값을 supports list 에 저장
        supports = [candidate[x] / self.tr_size for x in frequents]
        frequents = [[x] for x in frequents]
        return frequents, supports

    @staticmethod
    def _self_join(Lk, k):
        list_length = len(Lk)
        candidates = []
        for i in range(list_length):
            for j in range(i + 1, list_length):
                Lp = Lk[i][:k-1]
                Lq = Lk[j][:k-1]
                if Lp == Lq:
                    new_candidate = Lp + Lk[i][k-1:] + Lk[j][k-1:]
                    candidates.append(sorted(new_candidate))
        return candidates

    @staticmethod
    def _pruning(candidates, before_list):
        pruned = list(candidates)
        for c in candidates:
            for i in range(len(c)):
                subset = c[:i] + c[i+1:]
                try:
                    if subset not in before_list:
                        pruned.remove(c)
                        continue
                except ValueError:
                    continue
        return pruned

    @staticmethod
    def _is_contain(candidate, transaction):
        for item in candidate:
            if item not in transaction:
                return False
        return True

    @staticmethod
    def _itemset_format(itemset):
        return '{' + ",".join(map(str, itemset)) + '}'

    @staticmethod
    def _all_subsets(ss):
        return chain(*map(lambda x: combinations(ss, x), range(0, len(ss) + 1)))

    @staticmethod
    def _round(num, n):
        return floor(num * (10**n) + 0.5) / (10**n)

    def _generate_association_rules(self, L, supports):
        _L = [x for Lk in L for x in Lk]
        _support = [x for sk in supports for x in sk]

        output = open(self.out_path, 'w')

        for items in _L:
            if len(items) < 2:
                continue
            for subset in self._all_subsets(items):
                subset = list(subset)
                if len(subset) < 1 or set(subset) == set(items):
                    continue
                associative_item = sorted(list(set(items) - set(subset)))
                support = _support[_L.index(items)]
                confidence = support / _support[_L.index(subset)]

                item_set_str = self._itemset_format(subset)
                associative_item_str = self._itemset_format(associative_item)

                outstr = '%s\t%s\t%.2f\t%.2f\n' % (item_set_str, associative_item_str, self._round(support*100, 2), self._round(confidence*100, 2))
                output.write(outstr)
        output.close()

    def run(self):
        L1, sup1 = self._get_frequent_information_size_1()
        L = [L1]
        C = [L[0]]
        support = [sup1]
        k = 0

        while L[k]:
            C.append(self._self_join(L[k], k + 1))
            C[k + 1] = self._pruning(C[k + 1], L[k])
            count = [0] * len(C[k + 1])

            for tr in self._transaction_generator():
                for i in range(len(C[k + 1])):
                    if self._is_contain(C[k + 1][i], tr):
                        count[i] += 1

            new_frequent = []
            support_k = []
            for i in range(len(C[k + 1])):
                if count[i] / self.tr_size >= self.min_sup:
                    new_frequent.append(C[k + 1][i])
                    support_k.append(count[i] / self.tr_size)

            support.append(support_k)
            L.append(new_frequent)
            k += 1

        self._generate_association_rules(L, support)
        return L, support

if __name__ == '__main__':
    try:
        if len(sys.argv) != 4:
            raise ValueError
    except ValueError:
        print('Arguments must consist of min-support, input file name and output file name')
    apriori = Apriori(int(sys.argv[1]), sys.argv[2], sys.argv[3])
    apriori.run()




