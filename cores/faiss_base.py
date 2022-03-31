# -*- coding: utf-8 -*-

import os
import math

import faiss
import numpy as np
from pydantic import validate_arguments

from el_logging import logger

from ..helpers import utils
from ..helpers import validator


class FaissBase:
    """A base class to provide base APIs of FAISS library.
    FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.

    Attributes:
        [static] MAX_N_NEIGHBORS   (int          ) [1000 ]: Possible maximum number of similar vectors to search.
        [static] LOW_LIM_N_MIN_VEC (int          ) [100  ]: Lower limit for number of minimum 'db_vectors' to train k-means model.
        [static] LOW_LIM_N_ITER    (int          ) [10   ]: Lower limit for number of iteration to train k-means.
        [static] LOW_LIM_N_REDO    (int          ) [1    ]: Lower limit for number of redo to train k-means.
        [static] LOW_LIM_K         (int          ) [1    ]: Lower limit for K (number of centroids).

        [object] db_vectors        (np.ndarray   )        : Base DB vectors for training k-means.
        [object] db_labels         (list         )        : DB vectors cluster labels after training k-means.
        [object] index             (faiss.swig...)        : Faiss index for searching similar vectors.
        [object] kmeans            (faiss.Kmeans )        : Faiss k-means model.
        [object] max_n_neighbors   (int          ) [1000 ]: Maximum number of possible neighbor vectors to search.
        [object] min_n_vectors     (int          )        : Minimum number of 'db_vectors' to train k-means model.
        [object] k                 (int          )        : K (number of centroids) for k-means model.
        [object] n_redo            (int          )        : Number of redo for training k-means model.
        [object] n_iter            (int          )        : Number of iteration for training k-means model.
        [object] n_cells           (int          )        : Number cells (centers) for training faiss index to search faster.
        [object] gpu               (bool         ) [False]: Use GPU or CPU for training, clustering, and searching.
        [object] is_trained        (bool         ) [False]: Checker variable for k-means model is trained or not.
        [object] verbose           (bool         ) [False]: Print more detailed information about training and others.

    Methods:
        [object] add_vectors()           : Add new vectors to 'db_vectors'.
        [object] _del_duplicate_vectors(): Delete duplicated vectors before adding new vectors to 'db_vectors'.
        [object] search_similars()       : Search similar vectors to remove duplicated vectors.
        [object] _cal_n_search_cells()   : Calculate number of search cells for searchinig similar vectors.
        [object] _train_index()          : Train faiss index based on current 'db_vectors'.
        [object] _cal_n_cells()          : Calculate number of cells for training faiss index.
        [object] train_kmeans()          : Train k-means model based on 'db_vectors' and other parameters.
        [object] is_db_vectors_enough()  : Check 'db_vectors' is enough to train k-means model.
        [object] cluster()               : Cluaster vectors and return cluster ids as list.
    """

    MAX_N_NEIGHBORS = 1000
    LOW_LIM_N_MIN_VEC = 100
    LOW_LIM_N_ITER = 10
    LOW_LIM_N_REDO = 1
    LOW_LIM_K = 1


    def __init__(self, db_vectors: np.ndarray=None, min_n_vectors: int=1000, k: int=2, n_redo: int=4, n_iter: int=40, gpu: bool=False, verbose: bool=False):
        """FaissBase constructor method.

        Args:
            db_vectors    (np.ndarray, optional) [None ]: Base DB vectors for training k-means.
            min_n_vectors (int       , optional) [1000 ]: Minimum number of 'db_vectors' to train k-means model.
            k             (int       , optional) [2    ]: K (number of centroids) for k-means model.
            n_redo        (int       , optional) [4    ]: Number of redo for training k-means model.
            n_iter        (int       , optional) [40   ]: Number of iteration for training k-means model.
            gpu           (bool      , optional) [False]: Use GPU or CPU for training, clustering, and searching.
            verbose       (bool      , optional) [False]: Print more detailed information about training and others.
        """

        self.min_n_vectors = min_n_vectors
        self.k = k
        self.n_iter = n_iter
        self.n_redo = n_redo
        self.gpu = gpu


        self.max_n_neighbors = FaissBase.MAX_N_NEIGHBORS

        if os.getenv('DEBUG') == 'true':
            verbose = True
        self.verbose = verbose

        if not validator.is_empty(db_vectors):
            self.add_vectors(db_vectors)


    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def add_vectors(self, new_vectors: np.ndarray):
        """Add new vectors to 'db_vectors'.

        Args:
            new_vectors (np.ndarray, required): New vectors to add 'db_vectors'.

        Returns:
            bool: True when added new vectors to 'db_vectors', False when it couldn't.
        """

        # new_vectors = self._del_duplicate_vectors(new_vectors)
        # if validator.is_empty(new_vectors):
        #     logger.warning("Vectors are duplicated or already exists, can not add vectors!")
        #     return False

        try:
            if validator.is_empty(new_vectors):
                raise ValueError("'new_vectors' argument numpy array is empty!")

            if new_vectors.ndim == 1:
                new_vectors = np.expand_dims(new_vectors, 0)
            elif new_vectors.ndim != 2:
                raise ValueError(f"'new_vectors' argument numpy array dimension [{new_vectors.ndim}D] is invalid, should be [2D]!")
        except ValueError as err:
            logger.error(err)
            raise

        if new_vectors.dtype != 'float32':
            new_vectors = new_vectors.astype('float32')

        if validator.is_empty(self.db_vectors):
            self.db_vectors = new_vectors
        else:
            self.db_vectors  = np.concatenate((self.db_vectors , new_vectors))

        self._train_index()

        if validator.is_empty(self.db_labels):
            self.db_labels = []
        self.db_labels.extend([{ "cluster_id": -1, "distance": None }] * len(new_vectors))
        return True


    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _del_duplicate_vectors(self, vectors: np.ndarray):
        """Delete duplicated vectors before adding new vectors to 'db_vectors'.

        Args:
            vectors (np.ndarray, required): Vectors to check and remove duplicated vectors.

        Returns:
            np.ndarray: Only unique vectors.
        """
        try:
            if validator.is_empty(vectors):
                raise ValueError("'vectors' argument numpy array is empty!")

            if vectors.ndim == 1:
                vectors = np.expand_dims(vectors, 0)
            elif vectors.ndim != 2:
                raise ValueError(f"'vectors' argument numpy array dimension [{vectors.ndim}D] is invalid, should be [2D]!")
        except ValueError as err:
            logger.error(err)
            raise

        if vectors.dtype != 'float32':
            vectors = vectors.astype('float32')
        vectors = np.unique(vectors, axis=0)

        if validator.is_empty(self.db_vectors):
            return vectors

        _indices, _distances = self.search_similars(vectors)
        _dup_ids = []
        for _row, _dist in enumerate(_distances[:, 0]):
            if _dist == 0:
                _dup_ids.append(_row)
        return np.delete(vectors, _dup_ids, 0)


    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def search_similars(self, search_vectors: np.ndarray, n_neighbors: int=1, n_search_cells: int=None):
        """Search similar vectors to remove duplicated vectors.

        Args:
            search_vectors (np.ndarray, required)       : Searching vectors from 'db_vectors' (faiss index).
            n_neighbors    (int       , optional) [1   ]: Number of similar neighbors (vectors) to search.
            n_search_cells (int       , optional) [None]: Number of search cells from faiss index, searching will become slower if it's increase.

        Returns:
            tuple(np.ndarray, np.ndarray): Searched vectors indices with distances based on 'db_vectors'.
        """

        try:
            if validator.is_empty(self.db_vectors):
                raise RuntimeError("'db_vectors' is empty, should add vectors with 'add_vector()' function!")

            if validator.is_empty(self.index):
                raise RuntimeError("'index' is empty, should train with 'train()' function!")
        except RuntimeError as err:
            logger.error(err)
            raise

        try:
            if validator.is_empty(search_vectors):
                raise ValueError("'search_vectors' argument numpy array is empty!")

            if search_vectors.ndim == 1:
                search_vectors = np.expand_dims(search_vectors, 0)
            elif search_vectors.ndim != 2:
                raise ValueError(f"'search_vectors' argument numpy array dimension [{search_vectors.ndim}D] is invalid, should be [2D]!")

            if search_vectors.shape[1] != self.db_vectors.shape[1]:
                raise ValueError(f"'search_vectors' argument vector dimension [{search_vectors.shape[1]}D] is invalid, should be [{self.db_vectors.shape[1]}D]!")

            if search_vectors.dtype != 'float32':
                search_vectors = search_vectors.astype('float32')

            if self.db_vectors.shape[0] < self.max_n_neighbors:
                self.max_n_neighbors = self.db_vectors.shape[0]

            if (n_neighbors <= 0) or (self.max_n_neighbors < n_neighbors):
                raise ValueError(f"'n_neighbors' argument value '{n_neighbors}' is invalid, should be between '0' < and <= '{self.max_n_neighbors}'!")

            if validator.is_empty(n_search_cells):
                n_search_cells = self._cal_n_search_cells()

            if (n_search_cells <= 0) or (self.n_cells < n_search_cells):
                raise ValueError(f"'n_search_cells' argument value '{n_search_cells}' is invalid, should be between '0' < and <= '{self.n_cells}'!")
        except ValueError as err:
            logger.error(err)
            raise

        logger.debug('Searching similar vectors...')
        self.index.nprobe = n_search_cells
        _distances, _indices = self.index.search(search_vectors, n_neighbors)
        logger.debug('Searched similar vectors.')
        return _indices, _distances


    def _cal_n_search_cells(self):
        """Calculate number of search cells for searchinig similar vectors.

        Returns:
            int: Calculated number of search cells based on self.n_cells.
        """

        try:
            if validator.is_empty(self.n_cells):
                raise RuntimeError("'n_cells' is empty, should add vectors and use _cal_n_cells() method!")
        except RuntimeError as err:
            logger.error(err)
            raise

        _n_search_cells = int(math.sqrt(self.n_cells))
        if _n_search_cells == 0:
            _n_search_cells = 1
        return _n_search_cells


    @validate_arguments()
    def _train_index(self, n_cells: int=None, metric: int=faiss.METRIC_L2):
        """Train faiss index based on current 'db_vectors'.

        Args:
            n_cells (int, optional) [None           ]: Number of cells (centers) to train faiss index.
            metric  (int, optional) [faiss.METRIC_L2]: Faiss metric ID number.

        Returns:
            bool: True when faiss index is trained, False for not.
        """

        if validator.is_empty(n_cells):
            self.n_cells = self._cal_n_cells()
        else:
            self.n_cells = n_cells

        if not validator.is_empty(self.index):
            del self.index

        logger.debug('Training faiss index...')
        _vector_dim = self.db_vectors.shape[1]
        _quantizer = faiss.IndexFlatL2(_vector_dim)
        _index_ivf = faiss.IndexIVFFlat(_quantizer, _vector_dim, self.n_cells, metric)
        if self.gpu:
            # logger.debug(f'Number of available GPUs = {faiss.get_num_gpus()}')
            self.index = faiss.index_cpu_to_all_gpus(_index_ivf)
        else:
            self.index = _index_ivf

        self.index.train(self.db_vectors)
        self.index.add(self.db_vectors)
        logger.debug(f'Number of total indices = {self.index.ntotal}')
        logger.debug('Trained faiss index.')
        return True


    def _cal_n_cells(self):
        """Calculate number of cells for training faiss index.

        Returns:
            int: Calculated number of cells based on 'db_vectors'.
        """

        try:
            if validator.is_empty(self.db_vectors):
                raise RuntimeError("'db_vectors' is empty, should add vectors with 'add_vector()' function!")
        except RuntimeError as err:
            logger.error(err)
            raise

        _n_cells = int(math.sqrt(self.db_vectors.shape[0] / 10))
        if _n_cells == 0:
            _n_cells = 1
        return _n_cells


    @validate_arguments()
    def train_kmeans(self, **kwargs):
        """Train k-means model based on 'db_vectors' and other parameters.

        Args:
            **kwargs (dict, optional): This class attributes as keyword arguments (dictionary).
                Example: { k=2, n_redo=4 }

        Returns:
            bool: True when k-means model is trained, False for not.
        """

        for _key, _val in kwargs.items():
            # Raise AttributeError when there is no such '{_key}' attribute
            getattr(self, _key)
            setattr(self, _key, _val)

        try:
            if validator.is_empty(self.db_vectors):
                raise RuntimeError("'db_vectors' is empty, should add vectors with 'add_vector()' function!")

            if not self.is_db_vectors_enough():
                raise RuntimeError(f"'db_vectors' vector size [{self.db_vectors.shape[0]}] is not enough, should be at least [{self.min_n_vectors}] vectors!")
        except RuntimeError as err:
            logger.error(err)
            raise

        logger.debug('Training k-means model...')
        _vector_dim = self.db_vectors.shape[1]
        if not validator.is_empty(self.kmeans):
            delattr(self, 'kmeans')
        self.kmeans = faiss.Kmeans(_vector_dim, self.k, nredo=self.n_redo, niter=self.n_iter, gpu=self.gpu, verbose=self.verbose)
        self.kmeans.train(self.db_vectors)
        self.is_trained = True

        _db_labels = []
        _cluster_ids, _distances = self.cluster(self.db_vectors)
        _distances = _distances.tolist()
        for _i in range(len(_cluster_ids)):
            _label_dict = {
                "cluster_id": _cluster_ids[_i],
                "distance": _distances[_i]
            }
            _db_labels.append(_label_dict)
        self.db_labels = _db_labels
        logger.debug('Trained k-means model.')
        return True


    def is_db_vectors_enough(self):
        """Check 'db_vectors' is enough to train k-means model.

        Returns:
            bool: True when it's possible to train, False for not.
        """

        if not validator.is_empty(self.db_vectors) and (self.min_n_vectors <= self.db_vectors.shape[0]):
            return True
        return False


    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def cluster(self, vectors: np.ndarray):
        """Cluster vectors and return cluster ids as list.

        Args:
            vectors (np.ndarray): Clustering vectors.

        Returns:
            tuple(list, np.ndarray): Return cluster IDs and distances from centroids as tuple.
        """

        try:
            if validator.is_empty(vectors):
                raise ValueError("'vectors' argument numpy array is empty!")

            if vectors.ndim == 1:
                vectors = np.expand_dims(vectors, 0)
            elif vectors.ndim != 2:
                raise ValueError(f"'vectors' argument numpy array dimension [{vectors.ndim}D] is invalid, should be [2D]!")
        except ValueError as err:
            logger.error(err)
            raise

        if vectors.dtype != 'float32':
            vectors = vectors.astype('float32')

        _results = self.kmeans.index.search(vectors, 1)
        _cluster_ids = _results[1].reshape(-1).tolist()
        _distances = _results[0].reshape(-1)
        return _cluster_ids, _distances


    ### ATTRIBUTES ###
    ## db_vectors ##
    @property
    def db_vectors(self):
        try:
            return self.__db_vectors
        except AttributeError:
            return None

    @db_vectors.setter
    def db_vectors(self, db_vectors):
        try:
            if not isinstance(db_vectors, np.ndarray):
                raise TypeError(f"'db_vectors' argument type <{type(db_vectors).__name__}> is invalid, should be <np.ndarray>!")

            if validator.is_empty(db_vectors):
                raise ValueError("'db_vectors' argument numpy array is empty!")

            if db_vectors.ndim == 1:
                db_vectors = np.expand_dims(db_vectors, 0)
            elif db_vectors.ndim != 2:
                raise ValueError(f"'db_vectors' argument numpy array dimension [{db_vectors.ndim}D] is invalid, should be [2D]!")
        except Exception as err:
            logger.error(err)
            raise

        if db_vectors.dtype != 'float32':
            db_vectors = db_vectors.astype('float32')
        self.__db_vectors = db_vectors
    ## db_vectors ##


    ## db_labels ##
    @property
    def db_labels(self):
        try:
            return self.__db_labels
        except AttributeError:
            return None

    @db_labels.setter
    def db_labels(self, db_labels):
        try:
            if not isinstance(db_labels, list):
                raise TypeError(f"'db_labels' argument type <{type(db_labels).__name__}> is invalid, should be <list>!")
        except TypeError as err:
            logger.error(err)
            raise

        self.__db_labels = db_labels
    ## db_labels ##


    ## index ##
    @property
    def index(self):
        try:
            return self.__index
        except AttributeError:
            return None

    @index.setter
    def index(self, index):
        try:
            if index is None:
                raise TypeError(f"'index' argument value '{index}' is empty!")
        except TypeError as err:
            logger.error(err)
            raise

        self.__index = index

    @index.deleter
    def index(self):
        try:
            del self.__index
        except AttributeError:
            logger.warning("Not found any 'index' attribute to delete.")
    ## index ##


    ## kmeans ##
    @property
    def kmeans(self):
        try:
            return self.__kmeans
        except AttributeError:
            return None

    @kmeans.setter
    def kmeans(self, kmeans):
        try:
            if not isinstance(kmeans, faiss.Kmeans):
                raise TypeError(f"'kmeans' argument class {type(kmeans)} is invalid, should be <class 'faiss.Kmeans'> object!")
        except TypeError as err:
            logger.error(err)
            raise

        if hasattr(kmeans, 'index') and kmeans.index.is_trained:
            self.is_trained = True
        self.__kmeans = kmeans

    @kmeans.deleter
    def kmeans(self):
        try:
            del self.__kmeans
        except AttributeError:
            logger.warning("Not found any 'kmeans' attribute to delete.")
    ## kmeans ##


    ## max_n_neighbors ##
    @property
    def max_n_neighbors(self):
        try:
            return self.__max_n_neighbors
        except AttributeError:
            return FaissBase.MAX_N_NEIGHBORS

    @max_n_neighbors.setter
    def max_n_neighbors(self, max_n_neighbors):
        try:
            if not isinstance(max_n_neighbors, int):
                raise TypeError(f"'max_n_neighbors' argument type <{type(max_n_neighbors).__name__}> is invalid, should be <int>!")
        except TypeError as err:
            logger.error(err)
            raise

        self.__max_n_neighbors = max_n_neighbors
    ## max_n_neighbors ##


    ## min_n_vectors ##
    @property
    def min_n_vectors(self):
        try:
            return self.__min_n_vectors
        except AttributeError:
            return None

    @min_n_vectors.setter
    def min_n_vectors(self, min_n_vectors):
        try:
            if not isinstance(min_n_vectors, int):
                raise TypeError(f"'min_n_vectors' argument type <{type(min_n_vectors).__name__}> is invalid, should be <int>!")

            if min_n_vectors < self.LOW_LIM_N_MIN_VEC:
                raise ValueError(f"'min_n_vectors' argument value '{min_n_vectors}' is invalid, should be higher than >= '{self.LOW_LIM_N_MIN_VEC}'!")
        except Exception as err:
            logger.error(err)
            raise

        self.__min_n_vectors = min_n_vectors
    ## min_n_vectors ##


    ## k ##
    @property
    def k(self):
        try:
            return self.__k
        except AttributeError:
            return None

    @k.setter
    def k(self, k):
        try:
            if not isinstance(k, int):
                raise TypeError(f"'k' argument type <{type(k).__name__}> is invalid, should be <int>!")

            if k < FaissBase.LOW_LIM_K:
                raise ValueError(f"'k' argument value '{k}' is invalid, should be higher than > '{FaissBase.LOW_LIM_K}'!")
        except Exception as err:
            logger.error(err)
            raise

        self.__k = k
    ## k ##


    ## n_redo ##
    @property
    def n_redo(self):
        try:
            return self.__n_redo
        except AttributeError:
            return None

    @n_redo.setter
    def n_redo(self, n_redo):
        try:
            if not isinstance(n_redo, int):
                raise TypeError(f"'n_redo' argument type <{type(n_redo).__name__}> is invalid, should be <int>!")

            if n_redo < FaissBase.LOW_LIM_N_REDO:
                raise ValueError(f"'n_redo' argument value '{n_redo}' is invalid, should be higher than >= '{FaissBase.LOW_LIM_N_REDO}'!")
        except Exception as err:
            logger.error(err)
            raise

        self.__n_redo = n_redo
    ## n_redo ##


    ## n_iter ##
    @property
    def n_iter(self):
        try:
            return self.__n_iter
        except AttributeError:
            return None

    @n_iter.setter
    def n_iter(self, n_iter):
        try:
            if not isinstance(n_iter, int):
                raise TypeError(f"'n_iter' argument type <{type(n_iter).__name__}> is invalid, should be <int>!")

            if n_iter < FaissBase.LOW_LIM_N_ITER:
                raise ValueError(f"'n_iter' argument value '{n_iter}' is invalid, should be higher than >= '{FaissBase.LOW_LIM_N_ITER}'!")
        except Exception as err:
            logger.error(err)
            raise

        self.__n_iter = n_iter
    ## n_iter ##


    ## n_cells ##
    @property
    def n_cells(self):
        try:
            return self.__n_cells
        except AttributeError:
            return None

    @n_cells.setter
    def n_cells(self, n_cells):
        try:
            if not isinstance(n_cells, int):
                raise TypeError(f"'n_cells' argument type <{type(n_cells).__name__}> is invalid, should be <int>!")

            if validator.is_empty(self.db_vectors):
                raise RuntimeError("'db_vectors' is empty, should add vectors with 'add_vector()' function!")

            if (n_cells <= 0) or (self.db_vectors.shape[0] < n_cells):
                raise ValueError(f"'n_cells' argument value '{n_cells}' is invalid, should be betwee '0' < and <= '{self.db_vectors.shape[0]}'!")
        except Exception as err:
            logger.error(err)
            raise

        self.__n_cells = n_cells
    ## n_cells ##


    ## gpu ##
    @property
    def gpu(self):
        try:
            return self.__gpu
        except AttributeError:
            return False

    @gpu.setter
    def gpu(self, gpu):
        try:
            if not isinstance(gpu, bool):
                raise TypeError(f"'gpu' argument type <{type(gpu).__name__}> is invalid, should be <bool>!")
        except TypeError as err:
            logger.error(err)
            raise

        if gpu:
            _avail_num_gpus = faiss.get_num_gpus()
            if _avail_num_gpus == 0:
                logger.warning('Not found any GPUs, changing to CPU.')
                gpu = False
        self.__gpu = gpu
    ## gpu ##


    ## is_trained ##
    @property
    def is_trained(self):
        try:
            return self.__is_trained
        except AttributeError:
            return False

    @is_trained.setter
    def is_trained(self, is_trained):
        try:
            if not isinstance(is_trained, bool):
                raise TypeError(f"'is_trained' argument type <{type(is_trained).__name__}> is invalid, should be <bool>!")
        except TypeError as err:
            logger.error(err)
            raise

        if is_trained:
            if validator.is_empty(self.kmeans) or (not hasattr(self.kmeans, 'index')) or (not self.kmeans.index.is_trained):
                logger.warning("'kmeans' is not trained, changed 'is_trained' to 'False'!")
                is_trained = False
        else:
            if validator.is_empty(self.kmeans) and hasattr(self.kmeans, 'index') and self.kmeans.index.is_trained:
                is_trained = True
        self.__is_trained = is_trained
    ## is_trained ##


    ## verbose ##
    @property
    def verbose(self):
        try:
            return self.__verbose
        except AttributeError:
            return False

    @verbose.setter
    def verbose(self, verbose):
        try:
            if not isinstance(verbose, bool):
                raise TypeError(f"'verbose' argument type <{type(verbose).__name__}> is invalid, should be <bool>!")
        except TypeError as err:
            logger.error(err)
            raise

        self.__verbose = verbose
    ## verbose ##
    ### ATTRIBUTES ###


    ## METHOD OVERRIDING ##
    def __str__(self):
        _self_dict = utils.clean_obj_dict(self.__dict__, self.__class__.__name__)
        return f"{self.__class__.__name__}: {_self_dict}"

    def __repr__(self):
        return utils.obj_to_repr(self)
    ## METHOD OVERRIDING ##
