# -*- coding: utf-8 -*-

import os
import errno
import json

import faiss
import numpy as np
from scipy.spatial.distance import cdist
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from pydantic import validate_arguments

from el_logging import logger

from ..helpers import utils
from ..helpers import validator
from ..cores.faiss_base import FaissBase


class FaissKmeans(FaissBase):
    """A core class of 'faiss_mod' module to use faiss k-means model with file management.

    Parent class: FaissBase

    Attributes:
        [static] FAISS_FILES_SUFFIX (str ) ['faiss']: Suffix name for k-means model files.
        [static] LOW_LIM_MAX_K      (int ) [10     ]: Lower limit for maximum number of K (centroids).
        [static] UP_LIM_MAX_K       (int ) [25     ]: Upper limit for maximum number of K (centroids).

        [object] model_name         (str )          : Model name of k-means model files.
        [object] model_dir          (str )          : Directory path to read and save k-means model files.
        [object] model_files_meta   (dict)          : K-means model files metadata as dictionary.

    Methods:
        [static] is_model_files_exist()     : Checker method for model files exist or not.
        [static] _is_db_vector_files_exist(): Checker method for db vector files exist or not.
        [static] _is_meta_files_exist()     : Checker method for meta files exist or not.
        [static] _get_files_meta()          : Get k-means model files metadata.

        [object] load()                     : Load k-means model files and DB vector files to memory.
        [object] _load_db_vectors()         : Load DB vector files to memory.
        [object] add_vectors()              : Add new vectors to 'db_vectors'.
        [object] delete()                   : Delete k-means model files.
        [object] _delete_db_vectors()       : Delete DB vector files.
        [object] _delete_meta_files()       : Delete meta files.
        [object] save(save_db_vectors)      : Save k-means model files.
        [object] _save_db_vectors()         : Save DB vector files.
        [object] _save_meta_files()         : Save meta files.
        [object] train_auto_k()             : Train k-means model without K (Search suboptimal K automatically).
        [object] _get_elbow_k()             : Searching K by elbow method.
        [object] _get_silhouette_k()        : Searching K by silhouette method.
        [object] train_kmeans()             : Train k-means model.
    """

    FAISS_FILES_SUFFIX = 'faiss'
    LOW_LIM_MAX_K = 10
    UP_LIM_MAX_K = 25

    def __init__(self, model_name: str, model_dir: str, min_n_vectors: int=1000, k: int=2, n_redo: int=4, n_iter: int=40, gpu: bool=False, verbose: bool=False):
        """FaissKmeans constructor method.

        Args:
            model_name    (str , required)        : Model name of k-means model files.
            model_dir     (str , required)        : Directory path to read and save k-means model files.
            min_n_vectors (int , optional) [1000 ]: Minimum number of 'db_vectors' to train k-means model.
            k             (int , optional) [2    ]: K (number of centroids) for k-means model.
            n_redo        (int , optional) [4    ]: Number of redo for training k-means model.
            n_iter        (int , optional) [40   ]: Number of iteration for training k-means model.
            gpu           (bool, optional) [False]: Use GPU or CPU for training, clustering, and searching.
            verbose       (bool, optional) [False]: Print more detailed information about training and others.
        """

        super().__init__(min_n_vectors=min_n_vectors, k=k, n_redo=n_redo, n_iter=n_iter, gpu=gpu, verbose=verbose)
        self.model_name = model_name
        self.model_dir = model_dir
        self.model_files_meta = self._get_files_meta(self.model_name, self.model_dir)

        if self._is_db_vector_files_exist(self.model_name, self.model_dir):
            self._load_db_vectors()
            if self.is_model_files_exist(self.model_name, self.model_dir, check_db_vectors=False):
                self.load(load_db_vectors=False)


    ### STATIC METHODS ###
    @staticmethod
    @validate_arguments
    def is_model_files_exist(model_name: str, model_dir: str, check_db_vectors: bool=True):
        """Checker method for model files exist or not.

        Args:
            model_name       (str , required)       : Model name of k-means model files.
            model_dir        (str , required)       : Directory path to check k-means model files.
            check_db_vectors (bool, optional) [True]: Flag variable to check db_vector files also.

        Returns:
            bool: True when model files exist, False when doesn't exist.
        """

        _model_files_meta = FaissKmeans._get_files_meta(model_name, model_dir)
        if not os.path.isfile(_model_files_meta['kmeans_index_file_path']):
            return False

        if not os.path.isfile(_model_files_meta['centroids_file_path']):
            return False

        if check_db_vectors:
            return FaissKmeans._is_db_vector_files_exist(model_name, model_dir)
        return True


    @staticmethod
    @validate_arguments
    def _is_db_vector_files_exist(model_name: str, model_dir: str):
        """Checker method for db vector files exist or not.

        Args:
            model_name (str, required): Model name of db vector files.
            model_dir  (str, required): Directory path to check db vector files.

        Returns:
            bool: True when db vectors files exist, False when doesn't exist.
        """

        _model_files_meta = FaissKmeans._get_files_meta(model_name, model_dir)
        if not FaissKmeans._is_meta_files_exist(model_name, model_dir):
            return False

        if not os.path.isfile(_model_files_meta['index_file_path']):
            return False

        if not os.path.isfile(_model_files_meta['vector_file_path']):
            return False
        return True


    @staticmethod
    @validate_arguments
    def _is_meta_files_exist(model_name: str, model_dir: str):
        """Checker method for meta files exist or not.

        Args:
            model_name (str, required): Model name of meta files.
            model_dir  (str, required): Directory path to check meta files.

        Returns:
            bool: True when meta files exist, False when doesn't exist.
        """

        _model_files_meta = FaissKmeans._get_files_meta(model_name, model_dir)
        if not os.path.isfile(_model_files_meta['meta_file_path']):
            return False

        if not os.path.isfile(_model_files_meta['labels_file_path']):
            return False
        return True


    @staticmethod
    @validate_arguments
    def _get_files_meta(model_name: str, model_dir: str):
        """Get k-means model files metadata.

        Args:
            model_name (str, required): Model name of k-means model files.
            model_dir  (str, required): Directory path to read and save k-means model files.

        Returns:
            dict: K-means model files metadata as dictionary.
        """

        try:
            model_name = model_name.strip()
            if validator.is_empty(model_name):
                raise ValueError("'model_name' argument value is empty!")
            model_dir = model_dir.strip()
            if validator.is_empty(model_dir):
                raise ValueError("'model_dir' argument value is empty!")
        except ValueError as err:
            logger.error(err)
            raise

        _base_filename = f'{model_name}.{FaissKmeans.FAISS_FILES_SUFFIX}'

        _kmeans_index_filename = f'{_base_filename}.kmeans.index.bin'
        _kmeans_index_file_path = os.path.join(model_dir, _kmeans_index_filename)
        _centroids_filename = f'{_base_filename}.centroids.npy'
        _centroids_file_path = os.path.join(model_dir, _centroids_filename)
        _meta_filename = f'{_base_filename}.meta.json'
        _meta_file_path = os.path.join(model_dir, _meta_filename)
        _labels_filename = f'{_base_filename}.labels.json'
        _labels_file_path = os.path.join(model_dir, _labels_filename)
        _index_filename = f'{_base_filename}.index.bin'
        _index_file_path = os.path.join(model_dir, _index_filename)
        _vector_filename = f'{_base_filename}.vectors.npy'
        _vector_file_path = os.path.join(model_dir, _vector_filename)

        _model_files_meta = {
            'model_name': model_name,
            'model_dir': model_dir,
            'base_filename': _base_filename,
            'kmeans_index_filename': _kmeans_index_filename,
            'kmeans_index_file_path': _kmeans_index_file_path,
            'centroids_filename': _centroids_filename,
            'centroids_file_path': _centroids_file_path,
            'meta_filename': _meta_filename,
            'meta_file_path': _meta_file_path,
            'labels_filename': _labels_filename,
            'labels_file_path': _labels_file_path,
            'index_filename': _index_filename,
            'index_file_path': _index_file_path,
            'vector_filename': _vector_filename,
            'vector_file_path': _vector_file_path
        }
        return _model_files_meta
    ### STATIC METHODS ###


    ## Load methods ##
    @validate_arguments
    def load(self, load_db_vectors: bool=True):
        """Load k-means model files and DB vector files to memory.

        Args:
            load_db_vectors (bool, optional) [True]: Flag for load DB vector files or not.

        Returns:
            bool: True when successfully loaded k-means model files.
        """

        if load_db_vectors:
            self._load_db_vectors()

        try:
            if not os.path.isfile(self.model_files_meta['centroids_file_path']):
                raise RuntimeError(f"'{self.model_files_meta['centroids_file_path']}' file doesn't exist!")

            if not os.path.isfile(self.model_files_meta['kmeans_index_file_path']):
                raise RuntimeError(f"'{self.model_files_meta['kmeans_index_file_path']}' file doesn't exist!")
        except RuntimeError as err:
            logger.error(err)
            raise

        logger.info(f"Loading '{self.model_name}' kmeans model files...")
        try:
            logger.debug(f"Loading '{self.model_files_meta['centroids_file_path']}' centroids numpy file...")
            _vector_dim = self.db_vectors.shape[1]
            self.kmeans = faiss.Kmeans(_vector_dim, self.k, nredo=self.n_redo, niter=self.n_iter, gpu=self.gpu)
            self.kmeans.centroids = np.load(self.model_files_meta['centroids_file_path'])
            logger.debug(f"Successfully loaded '{self.model_files_meta['centroids_file_path']}' centroids numpy file.")

            logger.debug(f"Loading '{self.model_files_meta['kmeans_index_file_path']}' kmeans index file...")
            if self.gpu:
                _index = faiss.read_index(self.model_files_meta['kmeans_index_file_path'])
                self.kmeans.index = faiss.index_cpu_to_all_gpus(_index)
            else:
                self.kmeans.index = faiss.read_index(self.model_files_meta['kmeans_index_file_path'])
            self.is_trained = True
            logger.debug(f"Successfully loaded '{self.model_files_meta['kmeans_index_file_path']}' kmeans index file.")
        except Exception:
            logger.error(f"Failed to load '{self.model_name}' vector model files.")
            raise
        logger.success(f"Successfully loaded '{self.model_name}' kmeans model files.")
        return True


    def _load_db_vectors(self):
        """Load DB vector files to memory.

        Returns:
            bool: True when successfully loaded DB vectors.
        """

        try:
            if not os.path.isfile(self.model_files_meta['vector_file_path']):
                raise RuntimeError(f"'{self.model_files_meta['vector_file_path']}' file doesn't exist!")

            if not os.path.isfile(self.model_files_meta['index_file_path']):
                raise RuntimeError(f"'{self.model_files_meta['index_file_path']}' file doesn't exist!")

            if not os.path.isfile(self.model_files_meta['labels_file_path']):
                raise RuntimeError(f"'{self.model_files_meta['labels_file_path']}' file doesn't exist!")

            if not os.path.isfile(self.model_files_meta['meta_file_path']):
                raise RuntimeError(f"'{self.model_files_meta['meta_file_path']}' file doesn't exist!")
        except RuntimeError as err:
            logger.error(err)
            raise

        logger.info(f"Loading '{self.model_name}' DB vectors files...")
        try:
            logger.debug(f"Loading '{self.model_files_meta['vector_file_path']}' DB vectors numpy file...")
            self.db_vectors = np.load(self.model_files_meta['vector_file_path'])
            logger.debug(f"Successfully loaded '{self.model_files_meta['vector_file_path']}' DB vectors numpy file.")

            logger.debug(f"Loading '{self.model_files_meta['index_file_path']}' index file...")
            if self.gpu:
                _index = faiss.read_index(self.model_files_meta['index_file_path'])
                self.index = faiss.index_cpu_to_all_gpus(_index)
            else:
                self.index = faiss.read_index(self.model_files_meta['index_file_path'])
            logger.debug(f"Successfully loaded '{self.model_files_meta['index_file_path']}' index file.")

            logger.debug(f"Loading '{self.model_files_meta['labels_file_path']}' labels json file...")
            with open(self.model_files_meta['labels_file_path'], 'r') as _labels_json_file:
                self.db_labels = json.load(_labels_json_file)
            logger.debug(f"Successfully loaded '{self.model_files_meta['labels_file_path']}' labels json file.")

            logger.debug(f"Loading '{self.model_files_meta['meta_file_path']}' meta json file...")
            with open(self.model_files_meta['meta_file_path'], 'r') as _meta_json_file:
                _meta_json = json.load(_meta_json_file)
                self.k = _meta_json['k']
                self.n_redo = _meta_json['n_redo']
                self.n_iter = _meta_json['n_iter']
                self.gpu = _meta_json['gpu']
                self.min_n_vectors = _meta_json['min_n_vectors']
                self.n_cells = _meta_json['n_cells']
            logger.debug(f"Successfully loaded '{self.model_files_meta['meta_file_path']}' meta json file.")
        except Exception:
            logger.error(f"Failed to load '{self.model_name}' vector model files.")
            raise
        logger.success(f"Successfully loaded '{self.model_name}' DB vectors files.")
        return True
    ## Load methods ##


    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def add_vectors(self, new_vectors: np.ndarray):
        """Add new vectors to 'db_vectors'.

        Args:
            new_vectors (np.ndarray, required): New vectors to add 'db_vectors'.

        Returns:
            bool: True when successfully added new vectors to 'db_vectors', False when it couldn't.
        """

        if self.is_trained and self.is_db_vectors_enough():
            logger.warning(f"Already trained '{self.model_name}' k-means model, can not add more vectors!")
            return False

        logger.info(f"Adding new vectors to '{self.model_name}' model...")
        _is_added = super().add_vectors(new_vectors)
        if _is_added:
            if self._is_db_vector_files_exist(self.model_name, self.model_dir):
                self._delete_db_vectors()
            self._save_db_vectors()
            logger.success(f"Successfully added new vectors to '{self.model_name}' model.")
        else:
            return False
        return True


    ## Delete methods ##
    @validate_arguments
    def delete(self, delete_db_vectors: bool=True):
        """Delete k-means model files.

        Args:
            delete_db_vectors (bool, optional) [True]: Flag for delete DB vector files or not.

        Returns:
            bool: True when successfully deleted k-means model files.
        """

        try:
            if not os.path.isfile(self.model_files_meta['kmeans_index_file_path']):
                raise RuntimeError(f"'{self.model_files_meta['kmeans_index_file_path']}' kmeans index file doesn't exist!")

            if not os.path.isfile(self.model_files_meta['centroids_file_path']):
                raise RuntimeError(f"'{self.model_files_meta['centroids_file_path']}' centroids numpy file doesn't exist!")
        except RuntimeError as err:
            logger.error(err)
            raise

        logger.info(f"Deleting '{self.model_name}' kmeans model files...")
        try:
            logger.debug(f"Deleting '{self.model_files_meta['kmeans_index_file_path']}' kmeans index file...")
            os.remove(self.model_files_meta['kmeans_index_file_path'])
            logger.debug(f"Successfully deleted '{self.model_files_meta['kmeans_index_file_path']}' kmeans index file.")

            logger.debug(f"Deleting '{self.model_files_meta['centroids_file_path']}' centroids numpy file...")
            os.remove(self.model_files_meta['centroids_file_path'])
            logger.debug(f"Successfully deleted '{self.model_files_meta['centroids_file_path']}' centroids numpy file.")

            if delete_db_vectors:
                self._delete_db_vectors()
        except Exception:
            logger.error(f"Failed to delete '{self.model_name}' kmeans model files.")
            raise
        logger.success(f"Successfully deleted '{self.model_name}' kmeans model files.")
        return True


    def _delete_db_vectors(self):
        """Delete DB vector files.

        Returns:
            bool: True when successfully deleted DB vector files.
        """

        self._delete_meta_files()

        try:
            if not os.path.isfile(self.model_files_meta['index_file_path']):
                raise RuntimeError(f"'{self.model_files_meta['index_file_path']}' index file doesn't exist!")

            if not os.path.isfile(self.model_files_meta['vector_file_path']):
                raise RuntimeError(f"'{self.model_files_meta['vector_file_path']}' DB vectors numpy file doesn't exist!")
        except RuntimeError as err:
            logger.error(err)
            raise

        try:
            logger.debug(f"Deleting '{self.model_files_meta['index_file_path']}' index file...")
            os.remove(self.model_files_meta['index_file_path'])
            logger.debug(f"Successfully deleted '{self.model_files_meta['index_file_path']}' index file.")

            logger.debug(f"Deleting '{self.model_files_meta['vector_file_path']}' DB vectors numpy file...")
            os.remove(self.model_files_meta['vector_file_path'])
            logger.debug(f"Successfully deleted '{self.model_files_meta['vector_file_path']}' DB vectors numpy file.")
        except Exception:
            logger.error(f"Failed to delete '{self.model_name}' DB vectors files.")
            raise
        return True


    def _delete_meta_files(self):
        """Delete meta files.

        Returns:
            bool: True when successfully deleted meta files.
        """

        try:
            if not os.path.isfile(self.model_files_meta['meta_file_path']):
                raise RuntimeError(f"'{self.model_files_meta['meta_file_path']}' meta json file doesn't exist!")

            if not os.path.isfile(self.model_files_meta['labels_file_path']):
                raise RuntimeError(f"'{self.model_files_meta['labels_file_path']}' labels json file doesn't exist!")
        except RuntimeError as err:
            logger.error(err)
            raise

        try:
            logger.debug(f"Deleting '{self.model_files_meta['labels_file_path']}' labels json file...")
            os.remove(self.model_files_meta['labels_file_path'])
            logger.debug(f"Successfully deleted '{self.model_files_meta['labels_file_path']}' labels json file.")

            logger.debug(f"Deleting '{self.model_files_meta['meta_file_path']}' meta json file...")
            os.remove(self.model_files_meta['meta_file_path'])
            logger.debug(f"Successfully deleted '{self.model_files_meta['meta_file_path']}' meta json file.")
        except Exception:
            logger.error(f"Failed to delete '{self.model_name}' meta files.")
            raise
        return True
    ## Delete methods ##


    ## Save methods ##
    @validate_arguments
    def save(self, save_db_vectors: bool=True):
        """Save k-means model files.

        Args:
            save_db_vectors (bool, optional) [True]: Flag for saving DB vector files.

        Returns:
            bool: True when successfully saved k-means model files.
        """

        try:
            if not self.is_trained:
                raise RuntimeError(f"'{self.model_name}' kmeans is not trained, can not save model files!")

            if validator.is_empty(self.kmeans.index):
                raise RuntimeError(f"'{self.model_name}' kmeans index is empty, can not save kmeans index to file!")

            if validator.is_empty(self.kmeans.centroids):
                raise RuntimeError(f"'{self.model_name}' kmeans 'centroids' is empty, can not save centroids to numpy file!")
        except RuntimeError as err:
            logger.error(err)
            raise

        logger.info(f"Saving '{self.model_name}' kmeans model files...")
        try:
            logger.debug(f"Saving '{self.model_files_meta['kmeans_index_file_path']}' kmeans index file...")
            _index = self.kmeans.index
            if self.gpu:
                _index = faiss.index_gpu_to_cpu(self.kmeans.index)
            faiss.write_index(_index, self.model_files_meta['kmeans_index_file_path'])
            logger.debug(f"Successfully saved '{self.model_files_meta['kmeans_index_file_path']}' kmeans index file.")

            logger.debug(f"Saving '{self.model_files_meta['centroids_file_path']}' centroids numpy file...")
            np.save(self.model_files_meta['centroids_file_path'], self.kmeans.centroids)
            logger.debug(f"Successfully saved '{self.model_files_meta['centroids_file_path']}' centroids numpy file.")

            if save_db_vectors:
                self._save_db_vectors()
        except Exception:
            logger.error(f"Failed to save '{self.model_name}' kmeans model files.")
            raise
        logger.success(f"Successfully saved '{self.model_name}' kmeans model files.")
        return self.model_files_meta


    def _save_db_vectors(self):
        """Save DB vector files.

        Returns:
            bool: True when successfully saved DB vector files.
        """

        self._save_meta_files()

        try:
            if validator.is_empty(self.index):
                raise RuntimeError(f"'index' is empty, can not save index file!")

            if validator.is_empty(self.db_vectors):
                raise RuntimeError(f"'db_vectors' is empty, can not save DB vectors to numpy file!")
        except RuntimeError as err:
            logger.error(err)
            raise

        try:
            logger.debug(f"Saving '{self.model_files_meta['index_file_path']}' index file...")
            _index = None
            if self.gpu:
                _index = faiss.index_gpu_to_cpu(self.index)
            else:
                _index = self.index
            faiss.write_index(_index, self.model_files_meta['index_file_path'])
            logger.debug(f"Successfully saved '{self.model_files_meta['index_file_path']}' index file.")

            logger.debug(f"Saving '{self.model_files_meta['vector_file_path']}' DB vectors numpy file...")
            np.save(self.model_files_meta['vector_file_path'], self.db_vectors)
            logger.debug(f"Successfully saved '{self.model_files_meta['vector_file_path']}' DB vectors numpy file.")
        except Exception:
            logger.error(f"Failed to save '{self.model_name}' kmeans model files.")
            raise
        return True


    def _save_meta_files(self):
        """Save meta files.

        Returns:
            bool: True when successfully saved meta files.
        """

        try:
            if validator.is_empty(self.model_files_meta):
                raise RuntimeError(f"'model_files_meta' is empty, can not save meta json file!")

            if validator.is_empty(self.db_labels):
                raise RuntimeError(f"'db_labels' is empty, can not save labels to json file!")
        except RuntimeError as err:
            logger.error(err)
            raise

        try:
            logger.debug(f"Saving '{self.model_files_meta['meta_file_path']}' meta json file...")
            _meta_json = self.model_files_meta.copy()
            _meta_json['k'] = self.k
            _meta_json['n_redo'] = self.n_redo
            _meta_json['n_iter'] = self.n_iter
            _meta_json['gpu'] = self.gpu
            _meta_json['min_n_vectors'] = self.min_n_vectors
            _meta_json['n_db_vectors'] = self.db_vectors.shape[0]
            _meta_json['n_cells'] = self.n_cells
            with open(self.model_files_meta['meta_file_path'], 'w') as _meta_json_file:
                _meta_json_file.write(json.dumps(_meta_json, indent=4, ensure_ascii=False))
            logger.debug(f"Successfully saved '{self.model_files_meta['meta_file_path']}' meta json file.")

            logger.debug(f"Saving '{self.model_files_meta['labels_file_path']}' labels json file...")
            with open(self.model_files_meta['labels_file_path'], 'w') as _labels_json_file:
                _labels_json_file.write(json.dumps(self.db_labels, ensure_ascii=False))
            logger.debug(f"Successfully saved '{self.model_files_meta['labels_file_path']}' labels json file.")
        except Exception:
            logger.error(f"Failed to save '{self.model_name}' meta files.")
            raise
        return True
    ## Save methods ##


    ## Train methods ##
    @validate_arguments
    def train_auto_k(self, method: str='elbow', max_k: int=18):
        """Train k-means model without K (Search suboptimal K automatically).

        Args:
            method (str, optional) ['elbow']: Method ('elbow' or 'silho') for searching K.
            max_k  (int, optional) [18     ]: Maximum K range to search.

        Returns:
            bool: True when successfully trained k-means model.
        """

        method = method.lower().strip()
        try:
            if validator.is_empty(method):
                raise ValueError("'method' argument value is empty!")
        except ValueError as err:
            logger.error(err)
            raise

        if self.is_trained and self.is_db_vectors_enough():
            logger.warning(f"Already trained '{self.model_name}' k-means model, can not train again!")
            return False

        logger.info(f"Finding optimal K (number of centroids)...")
        _k = 2
        if method == 'elbow':
            _k = self._get_elbow_k(max_k)
        elif method == 'silho':
            _k = self._get_silhouette_k(max_k)
        else:
            try:
                raise ValueError(f"'method' argument value '{method}' is invalid, should be 'elbow' or 'silho'!")
            except ValueError as err:
                logger.error(err)
                raise
        logger.success(f"Found optimal K='{_k}'.")
        self.train_kmeans(k=_k)
        return True


    @validate_arguments
    def _get_elbow_k(self, max_k: int):
        """Searching K by elbow method.

        Args:
            max_k (int, required): Maximum K range to search.

        Returns:
            int: Found K (number of centroids).
        """

        try:
            if (max_k < self.LOW_LIM_MAX_K) or (self.UP_LIM_MAX_K < max_k):
                raise ValueError(f"'max_k' argument value is invalid '{max_k}', should bewtween '{self.LOW_LIM_MAX_K}' <= and <= '{self.UP_LIM_MAX_K}'!")

            if validator.is_empty(self.db_vectors):
                raise RuntimeError("'db_vectors' is empty, should add vectors with 'add_vector()' function!")

            if not self.is_db_vectors_enough():
                raise RuntimeError(f"'db_vectors' vector size [{self.db_vectors.shape[0]}] is not enough, should be at least [{self.min_n_vectors}] vectors!")
        except Exception as err:
            logger.error(err)
            raise

        _k_range = range(1, max_k + 1)
        _distortions = []
        for _k in _k_range:
            _vector_dim = self.db_vectors.shape[1]
            _kmeans = faiss.Kmeans(_vector_dim, _k, nredo=self.n_redo, niter=self.n_iter, gpu=self.gpu, verbose=self.verbose)
            _kmeans.train(self.db_vectors)
            _distortions.append(sum(np.min(cdist(self.db_vectors, _kmeans.centroids, 'euclidean'), axis=1)) / self.db_vectors.shape[0])
            del _kmeans

        _kn_locator = KneeLocator(list(_k_range), _distortions, S=1.0, curve='convex', direction='decreasing')
        _k = _kn_locator.knee.item()
        return _k


    @validate_arguments
    def _get_silhouette_k(self, max_k: int):
        """Searching K by silhouette method.

        Args:
            max_k (int, required): Maximum K range to search.

        Returns:
            int: Found K (number of centroids).
        """

        try:
            if (max_k < self.LOW_LIM_MAX_K) or (self.UP_LIM_MAX_K < max_k):
                raise ValueError(f"'max_k' argument value is invalid '{max_k}', should bewtween '{self.LOW_LIM_MAX_K}' <= and <= '{self.UP_LIM_MAX_K}'!")

            if validator.is_empty(self.db_vectors):
                raise RuntimeError("'db_vectors' is empty, should add vectors with 'add_vector()' function!")

            if not self.is_db_vectors_enough():
                raise RuntimeError(f"'db_vectors' vector size [{self.db_vectors.shape[0]}] is not enough, should be at least [{self.min_n_vectors}] vectors!")
        except Exception as err:
            logger.error(err)
            raise

        _silhouette_scores = []
        for _k in range(2, max_k + 1):
            _vector_dim = self.db_vectors.shape[1]
            _kmeans = faiss.Kmeans(_vector_dim, _k, nredo=self.n_redo, niter=self.n_iter, gpu=self.gpu, verbose=self.verbose)
            _kmeans.train(self.db_vectors)
            _pred_cluster_ids = _kmeans.index.search(self.db_vectors, 1)[1].reshape(-1).tolist()
            _silhouette_scores.append(silhouette_score(self.db_vectors, _pred_cluster_ids, metric='euclidean'))
            del _kmeans

        _k = _silhouette_scores.index(max(_silhouette_scores)) + 2
        return _k


    @validate_arguments
    def train_kmeans(self, **kwargs):
        """Train k-means model.

        Returns:
            bool: True when successfully trained k-means model.
        """

        for _key, _val in kwargs.items():
            # Raise AttributeError when there is no such '{_key}' attribute
            getattr(self, _key)
            setattr(self, _key, _val)

        if self.is_trained and self.is_db_vectors_enough():
            logger.warning(f"Already trained '{self.model_name}' k-means model, can not train again!")
            return False

        logger.info(f"Trainning '{self.model_name}' kmeans model...")
        super().train_kmeans()

        if self._is_meta_files_exist(self.model_name, self.model_dir):
            self._delete_meta_files()
        self._save_meta_files()

        if self.is_model_files_exist(self.model_name, self.model_dir, check_db_vectors=False):
            self.delete(delete_db_vectors=False)
        self.save(save_db_vectors=False)
        logger.success(f"Successfully trained '{self.model_name}' kmeans model.")
        return True
    ## Train methods ##


    ### ATTRIBUTES ###
    ## model_name ##
    @property
    def model_name(self):
        try:
            return self.__model_name
        except AttributeError:
            return None

    @model_name.setter
    def model_name(self, model_name):
        try:
            if not isinstance(model_name, str):
                raise TypeError(f"'model_name' argument type <{type(model_name).__name__}> is invalid, should be <str>!")

            model_name = model_name.strip()
            if validator.is_empty(model_name):
                raise ValueError("'model_name' argument value is empty!")
        except Exception as err:
            logger.error(err)
            raise

        self.__model_name = model_name
        if not validator.is_empty(self.model_files_meta):
            self.model_files_meta = self._get_files_meta(self.__model_name, self.model_dir)
    ## model_name ##


    ## model_dir ##
    @property
    def model_dir(self):
        try:
            return self.__model_dir
        except AttributeError:
            return None

    @model_dir.setter
    def model_dir(self, model_dir):
        try:
            if not isinstance(model_dir, str):
                raise TypeError(f"'model_dir' argument type <{type(model_dir).__name__}> is invalid, should be <str>!")

            model_dir = model_dir.strip()
            if validator.is_empty(model_dir):
                raise ValueError("'model_dir' argument value is empty!")
        except Exception as err:
            logger.error(err)
            raise

        if not os.path.isdir(model_dir):
            logger.warning(f"'{model_dir}' directory doesn't exist!")
            try:
                os.makedirs(model_dir)
            except OSError as err:
                if err.errno == errno.EEXIST:
                    logger.info(f"'{model_dir}' directory already exists!")
                else:
                    logger.error(f"Failed to create '{model_dir}' directory.")
                    raise
            logger.success(f"Successfully created '{model_dir}' directory!")

        self.__model_dir = model_dir
        if not validator.is_empty(self.model_files_meta):
            self.model_files_meta = self._get_files_meta(self.model_name, self.__model_dir)
    ## model_dir ##


    ## model_files_meta ##
    @property
    def model_files_meta(self):
        try:
            return self.__model_files_meta
        except AttributeError:
            return None

    @model_files_meta.setter
    def model_files_meta(self, model_files_meta):
        try:
            if not isinstance(model_files_meta, dict):
                raise TypeError(f"'model_files_meta' argument type <{type(model_files_meta).__name__}> is invalid, should be <dict>!")
        except TypeError as err:
            logger.error(err)
            raise

        self.__model_files_meta = model_files_meta
    ## model_files_meta ##
    ### ATTRIBUTES ###


    ### OVERRIDING METHODS ###
    def __str__(self):
        _self_dict = utils.clean_obj_dict(self.__dict__, self.__class__.__name__)
        _self_dict = utils.clean_obj_dict(_self_dict, FaissBase.__name__)
        return f"{self.__class__.__name__}: {_self_dict}"

    def __repr__(self):
        return utils.obj_to_repr(self)
    ### OVERRIDING METHODS ###
