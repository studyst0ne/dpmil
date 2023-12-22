# @Time : 2022/3/4 16:39
# @Author: ZWX
# @Email: 935721546@qq.com
# @File : RepInsSe.py


import math
import random
import warnings
import sklearn.cluster as skc
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from Cluster import DensityPeak
from dbCluster import dbCluster
from MILTool import get_pos_neg_instance

warnings.filterwarnings("ignore")


def dis_euclidean(para_arr1, para_arr2):
    """The eucildean distance, i.e.m $||para_arr1 - para_arr2||^2$
    @param:
        para_arr1:
            The given array, e.g., np.array([1, 2])
        para_arr2:
            The given array like para_arr1.
    @return
        A scalar.
    """
    return np.sqrt(np.sum((para_arr1 - para_arr2) ** 2))


def max_similarity(ins_set, para_ins, para_gamma=1):
    ret_dis = -np.inf
    for ins in ins_set:
        ret_dis = max(ret_dis, np.exp(-para_gamma * (dis_euclidean(ins, para_ins) ** 2)))
    return ret_dis


class RepInsSe:

    def __init__(self, train_bag_set, prop_can_ins):
        """

        :param train_bag_set: The train bag set.
        :param prop_can_ins:  The proportion of candidate instance.
        """
        self.train_bag_set = train_bag_set
        self.prop_can_ins = prop_can_ins
        self.num_tr_bags = self.train_bag_set.shape[0]  # The cardinality  of the train bag set.
        self.dimensions = len(train_bag_set[0][0][0]) - 1

    def get_rep_ins(self):
        """

        Representative instance selection process:
        Step 1: Select the cluster center in the negative instance space as
        the initial negative representative instance set;
        Step 2: According to the designed scoring strategy, select the highest score in each positive bag
        as the positive representative;
        Step 3: Optimize the positive representative instance set;
        Step 4: Select the instance with the highest score from the negative bag as the negative representative
        according to the optimized representative instance set.
        :return: The positive representative instance set and negative representative instance set.
        """
        # Step 1: Use density peak clustering to select cluster centers as initial negative representative instance set.
        all_pos_ins, _, all_neg_ins, _ = get_pos_neg_instance(self.train_bag_set)
        # initial_neg_ins_set = KMeans(math.ceil(self.num_tr_bags * 0.2)).fit(all_neg_ins).cluster_centers_
        # initial_neg_ins_set = DensityPeak(all_neg_ins[:1000], math.ceil(self.num_tr_bags * 0.1)).cluster_instance()
        # Step 1.1 : Label the initial negative representative instance set with 0.
        # initial_neg_ins_label = np.zeros((initial_neg_ins_set.shape[0], 1)).astype("int32")
        initial_neg_ins_set, neg_ins_num = dbCluster(all_neg_ins).cluster()
        initial_neg_ins_label = np.zeros((len(initial_neg_ins_set), 1)).astype("int32")

        # Step 2: Select positive representative instance set.
        pos_rep_ins_set = []  # positive representative instance set.
        # Step 2.1: Get the highest-scoring instance in each positive bag as a positive representative.
        # print(initial_neg_ins_set)
        # print("***")
        # print(neg_ins_num)
        for i in range(self.num_tr_bags):
            if self.train_bag_set[i, -1] == 1:
                pos_rep_ins_set.append(
                    self.__get_rep_ins_single_bag(self.train_bag_set[i, 0][:, :-1], 1, initial_neg_ins_set,
                                                  initial_neg_ins_label))
        pos_rep_ins_set = np.array(pos_rep_ins_set)
        pos_rep_ins_label = np.ones((pos_rep_ins_set.shape[0], 1)).astype('int32')
        # positive representative instances label.

        # Step 3: Optimizing positive representative instance set.
        dis_pos_pos = cdist(pos_rep_ins_set, pos_rep_ins_set)
        dis_pos_neg = cdist(pos_rep_ins_set, initial_neg_ins_set)

        dis_pos = dis_pos_pos.mean(axis=1) - dis_pos_neg.mean(axis=1)

        index_pos = []
        dis_pos_abs = []
        for i in range(dis_pos.shape[0]):
            if dis_pos[i] < 0:
                index_pos.append(i)
                dis_pos_abs.append(np.abs(dis_pos[i]))
        if index_pos:
            sorted_dis_pos_abs = np.argsort(dis_pos_abs)[::-1][:neg_ins_num]
            final_index = []
            for i in range(len(sorted_dis_pos_abs)):
                final_index.append(index_pos[sorted_dis_pos_abs[i]])
                final_index = []
                for j in range(len(sorted_dis_pos_abs)):
                    final_index.append(index_pos[sorted_dis_pos_abs[j]])
            pos_rep_ins_set = pos_rep_ins_set[final_index]
            pos_rep_ins_label = pos_rep_ins_label[final_index]
        else:
            final_index = np.argsort(dis_pos)[:neg_ins_num]
            pos_rep_ins_set = pos_rep_ins_set[final_index]
            pos_rep_ins_label = pos_rep_ins_label[final_index]

        # gaussian_si = np.zeros(len(pos_rep_ins_set))
        # for i in range(len(pos_rep_ins_set)):
        #     gaussian_si[i] = max_similarity(initial_neg_ins_set, pos_rep_ins_set[i])  # 计算正代表实例与负代表实例集相似度
        # po_ins_re_idx = np.argsort(gaussian_si)[:len(initial_neg_ins_set)]
        # pos_rep_ins_set_final = np.zeros((len(initial_neg_ins_set), self.dimensions))
        # for i in range(len(initial_neg_ins_set)):
        #     pos_rep_ins_set_final[i] = pos_rep_ins_set[po_ins_re_idx[i]]
        # pos_rep_ins_label = np.ones((len(initial_neg_ins_set), 1)).astype("int32")

        # Step 4: Select negative representative instance set.
        # neg_rep_ins_set = []
        # temp_label = 0
        # for i in range(self.num_tr_bags):
        #     if self.train_bag_set[i, -1] != 1:
        #         temp_label = self.train_bag_set[i, -1]
        #         neg_rep_ins_set.append(
        #             self.__get_rep_ins_single_bag(self.train_bag_set[i, 0][:, :-1], 0, pos_rep_ins_set,
        #                                           pos_rep_ins_label))
        # neg_rep_ins_set = np.array(neg_rep_ins_set)  # neg representative instance set.
        # # Label of negative representative instances.
        # neg_rep_ins_label = np.zeros((neg_rep_ins_set.shape[0], 1)).astype("int32")
        # neg_rep_ins_label[neg_rep_ins_label != 1] = temp_label[0]
        neg_rep_ins_set = np.array(initial_neg_ins_set)  # neg representative instance set.
        neg_rep_ins_label = initial_neg_ins_label

        return pos_rep_ins_set, pos_rep_ins_label, neg_rep_ins_set, neg_rep_ins_label

    def __get_rep_ins_single_bag(self, bag, bag_label, ins_set, ins_label):
        """

        :param bag: A single bag.
        :param bag_label: The label of the bag.
        :param ins_set: Dataset opposite to the label of the bag.
        :param ins_label: Opposite of bag label.
        :return: The highest scoring instance in the bag.
        """
        # Step 1: Randomly get candidate instances in each bag.
        num_can_ins = math.ceil(bag.shape[0] * self.prop_can_ins)
        # can_ins_index = random.sample(range(0, bag.shape[0]), num_can_ins)
        # can_ins_set = bag[can_ins_index]
        can_ins_set = DensityPeak(bag, num_can_ins).cluster_instance()
        # Step 2: Get the highest scoring candidate instance.
        # Step 2.1: Get the weight vector of each candidate instance.
        can_ins_weight_vectors = []
        for i in range(num_can_ins):
            SVM_model = SVC(kernel='linear', max_iter=1000).fit(np.concatenate((ins_set, [can_ins_set[i]])),
                                                                np.concatenate((ins_label, [[bag_label]])))
            can_ins_weight_vectors.append(SVM_model.coef_[0])
        can_ins_weight_vectors = np.array(can_ins_weight_vectors)
        # Step 2.2 : Calculate the score of each candidate instance,
        # and then select the instance with the highest score as the representative.
        weight_matrix = np.zeros((num_can_ins, num_can_ins)).astype('float32')

        for i in range(num_can_ins):
            for j in range(num_can_ins):
                weight_matrix[i][j] = np.dot(can_ins_weight_vectors[i], can_ins_set[j])

        weight_matrix_no_diag = weight_matrix[~np.eye(weight_matrix.shape[0], dtype=bool)].reshape(
            weight_matrix.shape[0], -1)
        weight_matrix_no_diag = -(np.sort(-weight_matrix_no_diag))

        # Step 2.3: Calculate the respective scores for each candidate instance and other instances.
        score_i_to_j = np.zeros((num_can_ins, num_can_ins)).astype('float32')
        for i_ins in range(num_can_ins):
            for j_ins in range(num_can_ins):
                if i_ins != j_ins:
                    temp_weight_i = weight_matrix[i_ins][j_ins]
                    temp_weight_j = weight_matrix[j_ins][i_ins]
                    if temp_weight_i > 0 and temp_weight_j > 0:
                        sort_i_to_j = weight_matrix_no_diag[i_ins].tolist().index(temp_weight_i)
                        sort_j_to_i = weight_matrix_no_diag[j_ins].tolist().index(temp_weight_j)
                        score_i_to_j[i_ins][j_ins] = 1 / ((sort_i_to_j + 1) * (sort_j_to_i + 1))
                        score_i_to_j[j_ins][i_ins] = 1 / ((sort_i_to_j + 1) * (sort_j_to_i + 1))
        # Step 2.4: Calculate the score for each candidate instance.
        ins_score = score_i_to_j.sum(axis=1)
        # Step 2.5: Select the candidate instance with the highest score as the representative.
        rep_ins = can_ins_set[ins_score.tolist().index(max(ins_score))]  # The representative instance.

        return rep_ins
