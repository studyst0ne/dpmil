import sklearn.cluster as skc
import numpy as np


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


class dbCluster:
    def __init__(self, para_ins, eps=1.75, min_samples=2):
        self.ins = para_ins
        self.eps = eps
        self.min_samples = min_samples

    def cluster(self):
        db = skc.DBSCAN(eps=self.eps, min_samples=self.min_samples)  # 初始化聚类模型
        db.fit(self.ins)  # 拟合数据
        db_label = db.labels_  # 获取聚类结果
        cluster_num = db_label.max() + 1  # 获取聚类簇数量
        if db_label.any() == 0:  # 如果都属于第0簇
            cluster_num = 1
        cluster_idx = []  # 有效簇索引
        min_cluster_ins_num = np.floor(len(self.ins) * 0.05)  # 簇内实例数阈值
        for i in range(cluster_num):
            eq_letter = np.where(db_label == i)  # 找到属于第i簇的负实例索引集
            if len(eq_letter[0]) >= min_cluster_ins_num:  # 如果簇内实例数量大于阈值
                cluster_idx.append(i)  # 记录当前有效簇索引
        cluster_num = len(cluster_idx)  # 更新簇数量
        rep_ins_set = []
        for i in range(cluster_num):
            eq_letter = np.where(db_label == cluster_idx[i])
            ave_ins = np.average(self.ins[eq_letter[0]], axis=0)
            temp_dis = []
            for ins in self.ins:
                temp_dis.append(dis_euclidean(ins, ave_ins))
            temp_sorted_dis_idx = np.argsort(temp_dis)
            rep_ins_set.append(self.ins[temp_sorted_dis_idx[0]])
        return rep_ins_set, cluster_num
