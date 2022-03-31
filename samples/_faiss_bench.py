#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import itertools

import numpy as np

from el_logging import logger

from faiss_mod import FaissBase


def get_list(start, n=5, ratio=2):
    return [start * ratio ** i for i in range(n)]


def generate_vectors(vector_dim, n_vectors, dtype='float32', use_seed=True, rand_seed=1234, use_noise=True):
    if use_seed:
        np.random.seed(rand_seed)
    vectors = np.random.random((n_vectors, vector_dim)).astype(dtype)
    if use_noise:
        vectors[:, 0] += np.arange(n_vectors) / 1000
    return vectors

def generate_datasets(comb_list):
    datasets_dict = {}
    for i, comb in enumerate(comb_list):
        data_dict = {}
        data_dict['id'] = i
        data_dict['vector_dim'] = comb[0]
        data_dict['n_vectors'] = comb[1]
        start = time.time()
        data_dict['vectors'] = generate_vectors(comb[0], comb[1])
        end = time.time()
        data_dict['time'] = end - start
        datasets_dict[f'{comb[0]}_{comb[1]}'] = data_dict
    return datasets_dict


if __name__ == '__main__':
    main_start = time.time()
    vector_dim_list = get_list(16, n=3, ratio=4)
    n_db_vectors_list = get_list(1000, n=4, ratio=10)
    n_search_vectors_list = get_list(1, ratio=10)
    n_cells_list = get_list(1, ratio=10)
    n_search_cells_list = get_list(1, ratio=4)
    n_neighbors_list = get_list(1, ratio=3)

    logger.info(f'Dimension of vectors: {vector_dim_list}')
    logger.info(f'Number of database vectors: {n_db_vectors_list}')
    logger.info(f'Number of search vectors: {n_search_vectors_list}')
    logger.info(f'Number of centroids: {n_cells_list}')
    logger.info(f'Number of search centroids: {n_search_cells_list}')
    logger.info(f'Number of nearest neighbors: {n_neighbors_list}')

    start = time.time()
    db_comb_list = list(itertools.product(vector_dim_list, n_db_vectors_list))
    logger.success(f'Size of generated database vectors: {len(db_comb_list)}')
    db_datasets = generate_datasets(db_comb_list)
    # logger.info(db_datasets)
    query_comb_list = list(itertools.product(vector_dim_list, n_search_vectors_list))
    logger.success(f'Size of generated search vectors: {len(query_comb_list)}')
    search_datasets = generate_datasets(query_comb_list)
    # logger.info(search_datasets)
    end = time.time()
    logger.info(f'Dataset generation time in seconds: {(end - start)}s')

    comb_list = list(itertools.product(vector_dim_list, n_db_vectors_list, n_search_vectors_list, n_cells_list, n_search_cells_list, n_neighbors_list))

    result_list = []
    for i, comb in enumerate(comb_list):
        vector_dim = comb[0]
        n_db_vectors = comb[1]
        n_search_vectors = comb[2]
        n_cells = comb[3]
        n_search_cells = comb[4]
        n_neighbors = comb[5]

        db_dataset = db_datasets[f'{vector_dim}_{n_db_vectors}']
        search_dataset = search_datasets[f'{vector_dim}_{n_search_vectors}']
        if (n_cells <= db_dataset['n_vectors']) and (n_search_cells <= n_cells) and (n_neighbors <= db_dataset['n_vectors']):
            result_dict = {}

            fs = FaissBase(db_dataset['vectors'])
            train_start = time.time()
            fs.train(n_cells)
            train_end = time.time()
            train_time = train_end - train_start

            search_start = time.time()
            indices, distances = fs.search_similars(search_dataset['vectors'], n_neighbors, n_search_cells)
            similar_vectors = fs.get_raw_vectors(indices)
            search_end = time.time()
            search_time = search_end - search_start

            result_dict['vector_dim'] = vector_dim
            result_dict['n_db_vectors'] = n_db_vectors
            result_dict['n_search_vectors'] = n_search_vectors
            result_dict['n_cells'] = n_cells
            result_dict['n_search_cells'] = n_search_cells
            result_dict['n_neighbors'] = n_neighbors
            result_dict['indices'] = indices
            result_dict['distances'] = distances
            result_dict['train_time'] = train_time
            result_dict['search_time'] = search_time

            logger.info(result_dict)

            result_list.append(result_dict)

    # logger.info(result_list)
    main_end = time.time()
    logger.info(f'All process time in seconds: {(main_end - main_start)}s')
    logger.success('Done.')
