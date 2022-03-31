# Faiss Module

* Faiss library
* Vector similarity search
* Kmeans clustering

---

## Prerequisites

* **Miniconda (v3)** and **Python (>= v3.7.11)** - [https://github.com/bybatkhuu/wiki/blob/main/manuals/installs/miniconda.md](https://github.com/bybatkhuu/wiki/blob/main/manuals/installs/miniconda.md)
* For **NVIDIA GPU**:
    * **NVIDIA GPU driver (at least >= v418.39)** - [https://github.com/bybatkhuu/wiki/blob/main/manuals/installs/nvidia-driver-linux.md](https://github.com/bybatkhuu/wiki/blob/main/manuals/installs/nvidia-driver-linux.md)
    * **NVIDIA CUDA (v10.1)** - [https://github.com/bybatkhuu/wiki/blob/main/manuals/installs/cuda-linux.md](https://github.com/bybatkhuu/wiki/blob/main/manuals/installs/cuda-linux.md)

## Download

Clone repository by git:

```bash
git clone https://github.com/bybatkhuu/faiss_mod.git
```

## Getting started

### 1. Installation

Install python dependencies:

```bash
cd faiss_mod
cat requirements.txt | xargs -n 1 -L 1 pip install --no-cache-dir

## CPU version:
conda install -y -c pytorch faiss-cpu=1.7.0

## GPU version:
conda install -y -c pytorch faiss-gpu=1.7.0 cudatoolkit=10.1 # for CUDA 10.1
## or for a specific CUDA version
conda install -y -c pytorch faiss-gpu=1.7.0 cudatoolkit=10.2 # for CUDA 10.2
```

### 2. Test sample code

```bash
python faiss_kmeans_sample.py
```

### 3. Import module

```python
from faiss_mod import FaissKmeans

## Sample code:
model_name = 'model_name'
model_dir = '/path/models'

## Can check before loading model from file to memory
if FaissKmeans.is_model_files_exist(model_name, model_dir):
    print(f"YES: '{model_dir}/{model_name}' model files exists.")
else:
    print(f"NO: '{model_dir}/{model_name}' model files doesn't exists.")

## Create or load kmeans model
fk = FaissKmeans(model_name, model_dir, min_n_vectors=1000)
if not fk.is_trained:
    if not fk.is_db_vectors_enough():
        fk.add_vectors(vectors)
        if fk.is_db_vectors_enough():
            fk.train_auto_k()
        else:
            print(f"Can't train on too small 'db_vectors', add more vectors!")
    else:
        fk.train_auto_k()

if fk.is_trained:
    cluster_ids = fk.cluster(vectors)
    print(f"Number of centroids: {fk.k}")
    print(f"Cluster IDs: {cluster_ids}")
else:
    print("Not trained yet...")
```

---

## References

* [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
* [https://github.com/facebookresearch/faiss/blob/master/INSTALL.md](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md)
