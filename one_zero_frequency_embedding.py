# -*- coding: utf-8 -*-
"""
This code is for one zero frequency embedding, which is used as an ablation study method for nonBEND

@author: Chengzhang Zhu (kevin.zhu.china@gmail.com)


"""

import numpy as np


def one_zero_frequency_embedding(data):
    # calculate the value frequency
    value_freq_dict = dict()
    for j in range(data.shape[1]):
        attribute = data[:, j]
        value_freq_dict[j] = dict()
        unique_value = np.unique(attribute)
        for value in unique_value:
            value_freq_dict[j][value] = len(np.where(attribute == value)[0])*1.0/len(attribute)

    # generate data embedding dictionary
    embedding_dict = dict()
    for j in range(data.shape[1]):
        one_zero_dict = dict()
        unique_value = np.unique(data[:, j])
        # one zero embedding
        for value_index, value in enumerate(unique_value):
            one_zero_dict[value] = np.ones(len(unique_value))
            one_zero_dict[value][value_index] = 0
        # multiply exp(-frequency) and subspace probability
        for value in one_zero_dict:
            one_zero_dict[value] = one_zero_dict[value] * np.exp(-value_freq_dict[j][value])
        embedding_dict[j] = one_zero_dict

    # generate data embedding
    data_embedding_list = []
    for obj in data:
        data_embedding = []
        for j, value in enumerate(obj):
            data_embedding.append(embedding_dict[j][value])
        data_embedding_list.append(np.concatenate(data_embedding, axis=0))
    embedding = np.stack(data_embedding_list)
    return embedding
