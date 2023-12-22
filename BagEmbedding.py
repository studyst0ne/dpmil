# @Time : 2022/3/4 16:41
# @Author: ZWX
# @Email: 935721546@qq.com
# @File : BagEmbedding.py

import math
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from MILTool import normalize_vector
from RepInsSe import RepInsSe


def dis_euclidean(para_arr1, para_arr2):  # 求实例间距离
    """The eucildean distance, i.e.m $||para_arr1 - para_arr2||^2$
    @param:
        para_arr1:
            The given array, e.g., np.array([1, 2])
        para_arr2:
            The given array like para_arr1.
    @return
        A scalar.
    """
    return np.sqrt(np.sum((para_arr1 - para_arr2)**2))


class BagEmbedding:

    def __init__(self, train_bag_set, valid_bag_set, test_bag_set, prop_can_ins=0.2, isAblation=False, isDistance=False):
        self.test_bag_set = test_bag_set
        self.train_bag_set = train_bag_set
        self.valid_bag_set = valid_bag_set
        self.num_tr_bag_set = self.train_bag_set.shape[0]
        self.num_te_bag_set = self.test_bag_set.shape[0]
        self.num_va_bag_set = self.valid_bag_set.shape[0]
        self.prop_can_ins = prop_can_ins
        self.isAblation = isAblation
        self.distance = isDistance
        # print(self.isAblation)
        # Get the representative instance set.
        self.pos_rep_ins_set, self.pos_rep_ins_label, self.neg_rep_ins_set, self.neg_rep_ins_label = RepInsSe(
            self.train_bag_set, self.prop_can_ins).get_rep_ins()
        self.num_pos_rep = self.pos_rep_ins_set.shape[0]
        self.num_neg_rep = self.neg_rep_ins_set.shape[0]
        self.train_weight_vector = SVC(kernel='linear', max_iter=1000).fit(
            np.concatenate((self.pos_rep_ins_set, self.neg_rep_ins_set)),
            np.concatenate((self.pos_rep_ins_label, self.neg_rep_ins_label))).coef_[0]
        self.pos_mean, self.neg_mean = self.__get_mean_vector(self.train_bag_set)
        self.model = SVC(kernel='linear', max_iter=1000).fit(
            np.concatenate((self.pos_rep_ins_set, self.neg_rep_ins_set)),
            np.concatenate((self.pos_rep_ins_label, self.neg_rep_ins_label)))

    def get_train_embed_vectors(self):
        """
        Get the embedding vectors of the training bags.
        :return: The train embedding vectors.
        """
        # Step 1: Get the most positive (negative) representative instance corresponding to the train set.
        # pos_mean, neg_mean = self.__get_mean_vector(self.train_bag_set)
        train_embed_vectors = []
        train_embed_pos_vectors = []
        train_embed_neg_vectors = []
        train_embed_overall_vectors = []
        # Step 2: Iterate over each bag to get its embedding vector
        for i in range(self.num_tr_bag_set):
            train_embed_vectors.append(
                self.__get_embed_vectors(self.train_bag_set[i, 0][:, :-1]))
            # train_embed_vectors.append(
            #     self.get_vectors(self.train_bag_set[i, 0][:, :-1]))
            train_embed_pos_vectors.append(
                self.get_train_embed_pos_vectors(self.train_bag_set[i, 0][:, :-1])
            )
            train_embed_neg_vectors.append(
                self.get_train_embed_neg_vectors(self.train_bag_set[i, 0][:, :-1]))
            train_embed_overall_vectors.append(
                self.__get_overall_embed_vectors(self.train_bag_set[i, 0][:, :-1])
            )

        if self.isAblation:
            tr_p, tr_n, tr_b, tr_pn, tr_pb, tr_nb, tr_pnb = [], [], [], [], [], [], []
            for i in range(self.num_tr_bag_set):
                p, n, pn = self.__get_embed_vectors(self.train_bag_set[i, 0][:, :-1])
                tr_p.append(p)
                tr_n.append(n)
                tr_pn.append(pn)
            return np.array(tr_p), np.array(tr_n), np.array(tr_pn)

        return np.array(train_embed_vectors), np.array(train_embed_pos_vectors), np.array(train_embed_neg_vectors), np.array(train_embed_overall_vectors)

    def get_test_embed_vectors(self):
        """
        Get the embedding vectors of the test bags.
        :return: The test embedding vectors.
        """
        # Step 1: Get the most positive (negative) representative instance corresponding to the test set.
        # pos_mean, neg_mean = self.__get_mean_vector(self.train_bag_set)
        test_embed_vectors = []
        test_embed_pos_vectors = []
        test_embed_neg_vectors = []
        test_embed_overall_vectors = []
        # Step 2: Iterate over each bag to get its embedding vector
        for i in range(self.num_te_bag_set):
            test_embed_vectors.append(self.__get_embed_vectors(self.test_bag_set[i, 0][:, :-1]))
            # test_embed_vectors.append(self.get_vectors(self.test_bag_set[i, 0][:, :-1]))
            test_embed_pos_vectors.append(self.get_train_embed_pos_vectors(self.test_bag_set[i, 0][:, :-1]))
            test_embed_neg_vectors.append(self.get_train_embed_neg_vectors(self.test_bag_set[i, 0][:, :-1]))
            test_embed_overall_vectors.append(self.__get_overall_embed_vectors(self.test_bag_set[i, 0][:, :-1]))
        if self.isAblation:
            te_p, te_n, te_b, te_pn, te_pb, te_nb, te_pnb = [], [], [], [], [], [], []

            for i in range(self.num_te_bag_set):
                p, n, pn = self.__get_embed_vectors(self.test_bag_set[i, 0][:, :-1])
                te_p.append(p)
                te_n.append(n)
                te_pn.append(pn)
            return np.array(te_p), np.array(te_n), np.array(te_pn)

        return np.array(test_embed_vectors), np.array(test_embed_pos_vectors), np.array(test_embed_neg_vectors), np.array(test_embed_overall_vectors)

    def get_valid_embed_vectors(self):
        """
        Get the embedding vectors of the test bags.
        :return: The test embedding vectors.
        """
        # Step 1: Get the most positive (negative) representative instance corresponding to the test set.
        # pos_mean, neg_mean = self.__get_mean_vector(self.train_bag_set)
        valid_embed_vectors = []
        valid_embed_pos_vectors = []
        valid_embed_neg_vectors = []
        # Step 2: Iterate over each bag to get its embedding vector
        for i in range(self.num_va_bag_set):
            valid_embed_vectors.append(self.__get_embed_vectors(self.valid_bag_set[i, 0][:, :-1]))
            # valid_embed_vectors.append(self.get_vectors(self.valid_bag_set[i, 0][:, :-1]))
            valid_embed_pos_vectors.append(self.get_train_embed_pos_vectors(self.valid_bag_set[i, 0][:, :-1]))
            valid_embed_neg_vectors.append(self.get_train_embed_neg_vectors(self.valid_bag_set[i, 0][:, :-1]))
        if self.isAblation:
            te_p, te_n, te_b, te_pn, te_pb, te_nb, te_pnb = [], [], [], [], [], [], []

            for i in range(self.num_va_bag_set):
                p, n, pn = self.__get_embed_vectors(self.valid_bag_set[i, 0][:, :-1])
                te_p.append(p)
                te_n.append(n)
                te_pn.append(pn)
            return np.array(te_p), np.array(te_n), np.array(te_pn)

        return np.array(valid_embed_vectors), np.array(valid_embed_pos_vectors), np.array(valid_embed_neg_vectors)

    def __get_embed_vectors(self, bag):
        """
        Get the embedding vector for each bag
        :param bag: A single bag.
        :param pos_mean: The most positive representative instance.
        :param neg_mean: The most negative representative instance.
        :return: The bag embed vector.
        """
        # Step 1: Initialize the positive (negative) perspective vector.
        pos_pers_vector = np.zeros(bag.shape[1]).astype('float32')
        neg_pers_vector = np.zeros(bag.shape[1]).astype('float32')
        bag_pers_vector = self.get_bag_perspective_vector(bag=bag)

        # Step 2: Traverse the instances in each bag, divide it into two sub-bags,
        # and get the corresponding positive and negative perspective vectors.
        # pos_bag_dis = cdist(bag, self.pos_rep_ins_set)
        # neg_bag_dis = cdist(bag, self.neg_rep_ins_set)

        if self.distance:
            embed_vector = []
            for i in range(self.pos_rep_ins_set.shape[0]):
                dis_i_to_bag = []
                for j in range(bag.shape[0]):
                    dis_i_to_bag.append(math.exp(-(np.sqrt(sum(np.power((self.pos_rep_ins_set[i] - bag[j]), 2))))))
                embed_vector.append(max(dis_i_to_bag))

            for i in range(self.neg_rep_ins_set.shape[0]):
                dis_i_to_bag = []
                for j in range(bag.shape[0]):
                    dis_i_to_bag.append(math.exp(-(np.sqrt(sum(np.power((self.neg_rep_ins_set[i] - bag[j]), 2))))))
                embed_vector.append(max(dis_i_to_bag))
            return normalize_vector(np.array(embed_vector))

        for i in range(bag.shape[0]):

            prop = self.model.predict(bag[i].reshape(1, -1))

            if prop == 1:
                pos_pers_vector += bag[i] - self.pos_mean
            else:
                neg_pers_vector += bag[i] - self.neg_mean

        # Whether to perform ablation experiments.
        if not self.isAblation:
            return normalize_vector(np.concatenate((np.concatenate((pos_pers_vector, bag_pers_vector)), neg_pers_vector)))
            # return normalize_vector(np.concatenate((pos_pers_vector, neg_pers_vector)))
        else:
            return normalize_vector(pos_pers_vector), normalize_vector(neg_pers_vector), \
                   normalize_vector(np.concatenate((pos_pers_vector, neg_pers_vector)))

    def __get_overall_embed_vectors(self, bag):
        """
        Get the embedding vector for each bag
        :param bag: A single bag.
        :param pos_mean: The most positive representative instance.
        :param neg_mean: The most negative representative instance.
        :return: The bag embed vector.
        """
        # Step 1: Initialize the positive (negative) perspective vector.
        pos_pers_vector = np.zeros(bag.shape[1]).astype('float32')
        neg_pers_vector = np.zeros(bag.shape[1]).astype('float32')
        bag_pers_vector = self.get_bag_perspective_vector(bag=bag)

        # Step 2: Traverse the instances in each bag, divide it into two sub-bags,
        # and get the corresponding positive and negative perspective vectors.
        # pos_bag_dis = cdist(bag, self.pos_rep_ins_set)
        # neg_bag_dis = cdist(bag, self.neg_rep_ins_set)

        rep_ins_set = self.pos_rep_ins_set + self.neg_rep_ins_set
        if self.distance:
            embed_vector = []
            for i in range(rep_ins_set.shape[0]):
                dis_i_to_bag = []
                for j in range(bag.shape[0]):
                    dis_i_to_bag.append(math.exp(-(np.sqrt(sum(np.power((rep_ins_set[i] - bag[j]), 2))))))
                embed_vector.append(max(dis_i_to_bag))

            # for i in range(self.neg_rep_ins_set.shape[0]):
            #     dis_i_to_bag = []
            #     for j in range(bag.shape[0]):
            #         dis_i_to_bag.append(math.exp(-(np.sqrt(sum(np.power((self.neg_rep_ins_set[i] - bag[j]), 2))))))
            #     embed_vector.append(max(dis_i_to_bag))
            return normalize_vector(np.array(embed_vector))

        for i in range(bag.shape[0]):

            prop = self.model.predict(bag[i].reshape(1, -1))

            if prop == 1:
                pos_pers_vector += bag[i] - self.pos_mean
            else:
                neg_pers_vector += bag[i] - self.neg_mean

        # Whether to perform ablation experiments.
        if not self.isAblation:
            return normalize_vector(np.concatenate((np.concatenate((pos_pers_vector, bag_pers_vector)), neg_pers_vector)))
            # return normalize_vector(np.concatenate((pos_pers_vector, neg_pers_vector)))
        else:
            return normalize_vector(pos_pers_vector), normalize_vector(neg_pers_vector), \
                   normalize_vector(np.concatenate((pos_pers_vector, neg_pers_vector)))

    def __get_mean_vector(self, bag_set):
        """
        Get the most positive (negative) representative instance corresponding to the bag set.
        :param bag_set: The bag set.
        :return: The most positive (negative) representative instance.
        """
        index_ins_pos = []
        index_ins_neg = []
        # Step 1:Record the representative instance index that resulted in the distance (dis_pos or dis_neg) generation.
        for i in range(bag_set.shape[0]):
            # Step 1.1: Calculate the distance of the bag to the positive and negative representative sets.
            bag_pos_dis = cdist(bag_set[i, 0][:, :-1], self.pos_rep_ins_set)
            bag_neg_dis = cdist(bag_set[i, 0][:, :-1], self.neg_rep_ins_set)

            # Step 1.2: If it is a positive bag, find the closest distance value from the positive representative
            # and the farthest distance value from the negative representative.
            # else, find the farthest distance value from the positive representative
            # and the closest distance value from the negative representative.

            dis_pos = bag_pos_dis.min()
            dis_neg = bag_neg_dis.min()

            # # Step 1.3: Record the representative instance index that resulted in the distance (dis_pos) generation.
            for k in range(bag_pos_dis.shape[1]):
                for j in range(bag_pos_dis.shape[0]):
                    if bag_pos_dis[j][k] == dis_pos:
                        index_ins_pos.append(k)
                        break
            # # Step 1.4: Record the representative instance index that resulted in the distance (dis_neg) generation.
            for k in range(bag_neg_dis.shape[1]):
                for j in range(bag_neg_dis.shape[0]):
                    if bag_neg_dis[j][k] == dis_neg:
                        index_ins_neg.append(k)
                        break

        # Step 2: Find the representative instance within the index value
        # and average it to get the final most representative instance.
        pos_mean = self.pos_rep_ins_set[index_ins_pos].mean(axis=0)
        neg_mean = self.neg_rep_ins_set[index_ins_neg].mean(axis=0)

        return pos_mean, neg_mean

    def get_train_embed_pos_vectors(self, bag):
        pos_pers_vector = np.zeros(bag.shape[1]).astype('float32')
        for i in range(bag.shape[0]):
            prop = self.model.predict(bag[i].reshape(1, -1))
            if prop == 1:
                pos_pers_vector += bag[i] - self.pos_mean
        return normalize_vector(pos_pers_vector)

    def get_train_embed_neg_vectors(self, bag):
        neg_pers_vector = np.zeros(bag.shape[1]).astype('float32')
        for i in range(bag.shape[0]):
            prop = self.model.predict(bag[i].reshape(1, -1))
            if prop != 1:
                neg_pers_vector += bag[i] - self.neg_mean
        return normalize_vector(neg_pers_vector)

    def get_vectors(self, bag):
        global dis_i_to_bag_po, dis_i_to_bag_ne
        for i in range(self.pos_rep_ins_set.shape[0]):  # 遍历每一个正代表实例
            dis_i_to_bag_po = []
            for j in range(bag.shape[0]):  # 遍历包中每一个实例
                dis_i_to_bag_po.append(np.sqrt(sum(np.power((self.pos_rep_ins_set[i] - bag[j]), 2))))

        for i in range(self.neg_rep_ins_set.shape[0]):  # 遍历每一个负代表实例
            dis_i_to_bag_ne = []
            for j in range(bag.shape[0]):  # 遍历包中每一个实例
                dis_i_to_bag_ne.append(np.sqrt(sum(np.power((self.neg_rep_ins_set[i] - bag[j]), 2))))

        # print("正代表：", dis_i_to_bag_po)
        # print("负代表：", dis_i_to_bag_ne)
        result = [a_i - b_i for a_i, b_i in zip(dis_i_to_bag_po, dis_i_to_bag_ne)]
        # print("相减结果", result)
        # print("相减结果索引排序", np.argsort(result))
        result = np.argsort(result)
        zero_indices = np.where(result == 0)[0]  # 找到离正实例最近的
        # print(bag[0][:, :-1][zero_indices][0])
        indices = np.where(result == bag.shape[0]-1)[0]
        # print(bag[0][:, :-1][indices][0])

        pos_embed_vectors = np.zeros(bag.shape[1]).astype('float32')
        neg_embed_vectors = np.zeros(bag.shape[1]).astype('float32')
        # print(bag[zero_indices][0])
        # print(self.pos_rep_ins_set[0])
        for i in range(self.pos_rep_ins_set.shape[0]):
            pos_embed_vectors += bag[zero_indices][0] - self.pos_rep_ins_set[i]
        for i in range(self.neg_rep_ins_set.shape[0]):
            neg_embed_vectors += bag[indices][0] - self.neg_rep_ins_set[i]

        pos_embed_vectors = np.sign(pos_embed_vectors) * np.sqrt(np.abs(pos_embed_vectors))
        neg_embed_vectors = np.sign(neg_embed_vectors) * np.sqrt(np.abs(neg_embed_vectors))

        pos_embed_vectors = pos_embed_vectors / dis_euclidean(pos_embed_vectors, np.zeros_like(pos_embed_vectors))
        neg_embed_vectors = neg_embed_vectors / dis_euclidean(neg_embed_vectors, np.zeros_like(neg_embed_vectors))

        ret_embed_vectors = np.hstack((pos_embed_vectors, neg_embed_vectors))
        ret_embed_vectors = np.nan_to_num(ret_embed_vectors)
        return ret_embed_vectors

    def get_bag_perspective_vector(self, bag):
        score_set = []
        for i in range(bag.shape[0]):
            score_set.append(np.sum(bag[i] * self.train_weight_vector))
        return bag[np.argmax(score_set)]
