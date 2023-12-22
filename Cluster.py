import numpy as np
from scipy.spatial.distance import cdist


class DensityPeak:
    """
    Return only the cluster center instance.
    """

    def __init__(self, para_ins, para_num_center):
        self.ins = para_ins
        self.num_ins = self.ins.shape[0]
        self.num_center = para_num_center

    def cluster_instance(self):
        # The distance of instance.
        ins_dis = cdist(self.ins, self.ins)
        # The cut off distance.
        dis_cut = 0.4 * ins_dis.max()
        density_ins = np.zeros(self.num_ins).astype('float')
        for i_index in range(self.num_ins):
            if dis_cut == 0:
                density_ins[i_index] = 1
            else:
                density_ins[i_index] = sum(np.exp(-(ins_dis[i_index] / dis_cut) ** 2))

        distance_closest = []
        for i in range(self.num_ins):
            more_density_instance_index = []
            temp_density_instance = density_ins[i]
            for j in range(self.num_ins):
                if density_ins[j] > temp_density_instance:
                    more_density_instance_index.append(j)
            temp_distance_more_instance = []
            for index in range(0, len(more_density_instance_index)):
                index_k = more_density_instance_index[index]
                temp_distance_more_instance.append(ins_dis[i][index_k])
            if temp_distance_more_instance:
                temp_distance_more_instance.sort()
                distance_closest.append(temp_distance_more_instance[0])
            else:
                distance_closest.append(float('inf'))
        # Step 4. The lambda of discriminative instance.
        lambda_instance = np.multiply(distance_closest, self.num_ins).tolist()
        final_instance = []
        for i in range(self.num_center):
            index_most = lambda_instance.index(max(lambda_instance))
            final_instance.append(self.ins[index_most])
            lambda_instance[index_most] = -1
        return np.array(final_instance)
