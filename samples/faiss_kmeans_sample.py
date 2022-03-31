#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from sklearn.datasets import make_blobs

from el_logging import logger

from faiss_mod import FaissKmeans


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Faiss sample process module.')
    parser.add_argument('-n', '--model-name', dest='model_name', type=str, default='model_01', metavar='MODEL_NAME', help='K-means model name.')
    parser.add_argument('-d', '--model-dir', dest='model_dir', type=str, default='/home/batkhuu/workspaces/work/projects/faiss/faiss_api/cores/faiss_mod/models', metavar='MODEL_DIR', help='K-means model directory.')
    parser.add_argument('-D', '--vector-dim', dest='vector_dim', type=int, default='64', metavar='VECT_DIM', help='Dimension/length of vectors.')
    parser.add_argument('-N', '--n-db-vectors', dest='n_db_vectors', type=int, default='1000', metavar='NUM_VECT', help='Number of randomly generated database vectors.')
    parser.add_argument('-k', '--n-clusters', dest='n_clusters', type=int, default='4', metavar='K', help='Number of clusters (centroids).')
    parser.add_argument('-r', '--rand-seed', dest='rand_seed', type=int, default='123', metavar='RAN_SEED', help='Random generation seed.')
    parser.add_argument('-V', '--verbose', dest='verbose', default=False, action='store_true', help='Show detailed information.')
    parser.add_argument('-v', '--version', action='version', version='1.0', help='Shows version number of program and exit.')
    args = parser.parse_args()

    vectors, ground_truth = make_blobs(n_samples=args.n_db_vectors, centers=args.n_clusters, n_features=args.vector_dim, shuffle=True, random_state=args.rand_seed)

    model_name = args.model_name
    model_dir = args.model_dir

    ## Can check before loading model from file to memory
    if FaissKmeans.is_model_files_exist(model_name, model_dir):
        logger.info(f"YES: '{model_dir}/{model_name}' model files exists.")
    else:
        logger.info(f"NO: '{model_dir}/{model_name}' model files doesn't exists.")

    ## Create or load kmeans model
    fk = FaissKmeans(model_name, model_dir, min_n_vectors=1000, verbose=args.verbose)
    if not fk.is_trained:
        if not fk.is_db_vectors_enough():
            fk.add_vectors(vectors)
            if fk.is_db_vectors_enough():
                fk.train_auto_k()
            else:
                logger.warning(f"Can't train on too small 'db_vectors', add more vectors!")
        else:
            fk.train_auto_k()

    if fk.is_trained:
        cluster_ids = fk.cluster(vectors)
        logger.success(f"Number of centroids: {fk.k}")
        logger.info(f"Cluster IDs: {cluster_ids}")
    else:
        logger.warning("Not trained yet...")
