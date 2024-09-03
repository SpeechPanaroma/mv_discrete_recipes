import torch
import numpy as np
from joblib import Parallel, delayed
from basecluster import BaseCluster

class mvKMeansMinibatch(BaseCluster):
  r'''
  This class implements the optimized version of mvKMeansCluster. 
  It uses a mini-batch to perform all the operations to make it more efficient. 
  The core algorithm for multiview k-means clustering is same for both the class. 
  It handles data with two views for multiview clustering.
  
  Parameters
  ----------
  n_clusters: int, default: 2
              The number of clusters to form
  
  max_iter: int, default: 5
            The maximum number of iterations for the clustering algorithm.
  
  patience: int, default: 3
            The number of iterations to wait for convergence before stopping early.
  
  tol: float, default: 0.0001
       The tolerance for convergence, determining when to stop the algorithm.

  init: str, default: k-means++
        {Random or k-means++} The method for initialization of cluster centers for k-means 
        based clustering algorithm
  
  batch_size: int default: 4
              Mini batch-size
    
  n_jobs: int, default: -1
          The number of jobs to run in parallel. n_init will start parallelly with n_jobs
          workers.
  
  n_init: int, default: 1
          The number of times the algorithm will be run with different centroid seeds
  
  random_state: int, default: None 
                The seed used by the random number generator for reproducibility

  Pseudo Code
  -----------
    Unlabelled data D with 2 views and n samples
    Initialize cluster centers1θv
    (v ∈ {1, 2})
    m ← ⌊n/batch_size⌋
    Partition D | D = {b1, b2, ..., bm} where bj = D[j · batch_size : (j + 1) · batch_size]
    for b ∈ D do
      E step for view 2
    end for
    t ← 1
    L ← 0
    while stopping criterion on L is met do
      for b ∈ D do
        for v ∈ {1, 2} do
          M step view v: Find θvt that maximize the likelihood for
                          b given the expected values for hidden variables 
                          of view  ̄v of iteration t − 1
          E step view v: Compute expectation and loss (Lb) for hidden variables 
                          given the model parameters θvt
        end for
        L ← L + Lb
      end for
    end while
  '''
  def __init__(self, n_clusters=2, max_iter=5, patience=3, 
               tol=0.0001, init="random", batch_size=4, n_jobs=-1, 
               n_init=1,random_state = None):
    super().__init__(n_clusters=n_clusters, max_iter=max_iter, 
                     n_init=n_init, random_state=random_state)
    
    self.patience = patience
    self.tol = tol
    self.centroids = list()
    self.init = init
    self.batch_size = batch_size
    self.n_jobs = n_jobs
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.num_gpus = torch.cuda.device_count()

  def _compute_distance(self, X, centroids):
    r'''Compute the Euclidian distance between X and centroids
    
    Parameters
    ---------
    X: torch.Tensor
      shape (n_sample,n_features) a tensor presenting a single view
    
    centroids: torch.Tensor
                centroids for a single-view
    Returns
    -------
    dist: torch.Tensor
          Euclidian distance computed between X and centroids
    '''
    return torch.cdist(X,centroids)

  def _init_centroids(self,Xs):
    r''' initialize the centroid based on the given initialization method.
    Paramters
    ---------
    Xs : list of torch.Tensor
         - Xs length: n_views
         - Xs[i] shape: (n_samples, n_features_i)
    Returns
    -------
    centroids : list of torch.Tensor
            - centroids length: n_views
            - centroids[i] shape: (n_clusters, n_features_i)
    '''
    if self.init == "random":
      indices = np.random.choice(Xs[0].shape[0], self.n_clusters).tolist()
      centers1 = Xs[0][indices]
      centers2 = Xs[1][indices]
      centroids = [centers1, centers2]
    elif self.init == "k-means++":
      indices = list()
      centers2 = list()
      indices.append(int(torch.randint(0,Xs[1].shape[0],(1,))))
      centers2.append(Xs[1][indices[0], :])
      n_steps = int(Xs[0].shape[0]/self.batch_size)
      
      # Compute the remaining n_cluster centroids
      for cent in range(self.n_clusters - 1):
        dists = []
        for batch_step in range(n_steps):
          d = self._compute_distance(torch.stack(centers2),Xs[1][self.batch_size*batch_step:self.batch_size*(batch_step+1)])
          dists.append(torch.min(d, dim=0).values)
        dists = torch.stack(dists)
        max_index = torch.argmax(dists).item()
        indices.append(int(max_index))
        centers2.append(Xs[1][max_index])

      centers1 = Xs[0][indices]
      centers2 = torch.stack(centers2)
      centroids = [centers1, centers2]
    else:
      raise ValueError('Invalid initialization method.')
    return centroids

  def _em_step(self, X, partition, centroids):
    r'''This function computes one iteration of expectation-maximization.
    Parameters
    ----------
    X: torch.Tensor, shape (n_samples, n_features)
            tensor presenting a single view of the data.

    partition: torch.Tensor, shape (n_samples,)
            tensor of cluster labels indicating the cluster to which
            each data sample is assigned.

    centroids: torch.Tensor , shape (n_clusters, n_features)
            The current cluster centers.
    Returns
    -------
    new_parts: torch.Tensor, shape (n_samples,)
            The new cluster assignments for each sample in the data

    new_centers: torch.Tensor, shape (n_clusters, n_features)
            The updated cluster centers.

    o_funct: float
            The new value of the objective function.
    '''
    new_centers = list()
    n_samples = X.shape[0]
    for cl in range(self.n_clusters):
      # Recompute centroids using samples from each cluster
      mask = (partition == cl)
      if (torch.sum(mask) == 0):
        new_centers.append(centroids[cl])
      else:
        cent = torch.mean(X[mask], dim=0, dtype=float)
        new_centers.append(cent)

    new_centers = torch.stack(new_centers)
    # Compute expectation and objective function
    distances = self._compute_distance(X, new_centers)
    new_parts = torch.argmin(distances, dim=1).squeeze()
    min_dists = distances[torch.arange(n_samples), new_parts]
    o_funct = torch.sum(min_dists)
    return new_parts, new_centers, o_funct

  def _one_init (self,Xs):
    r'''Run the algorithm for one random initialization.

    Parameters
    ----------
    Xs : list of torch.Tensor
            - Xs length: n_views
            - Xs[i] shape: (n_samples, n_features_i)
    Returns
    -------
    intertia: int
              The final intertia for this run.

    centroids : list of torch.Tensor
            - centroids length: n_views
            - centroids[i] shape: (n_clusters, n_features_i)
    '''
    # Initialize centroids for clustering
    centroids = self._init_centroids(Xs)

    # Initializing partitions, objective value, and loop vars
    n_steps = int(Xs[0].shape[0]/self.batch_size)
    parts = []
    for batch_step in range(n_steps):
      distances = self._compute_distance(Xs[1][self.batch_size*batch_step:self.batch_size*(batch_step+1)], centroids[1])
      d = torch.argmin(distances, dim=1).squeeze()
      parts.append(d)
    parts = torch.cat(parts, dim =0)
    if self.num_gpus > 1 :
      partitions = [torch.zeros(parts.size()).to(torch.device('cuda:0')), parts]
      objective = [torch.tensor(float(10000000)).to(torch.device('cuda:0')), torch.tensor(float(10000000)).to(torch.device('cuda:1'))]
    else:
      partitions = [torch.zeros(parts.size()), parts]
      objective = [torch.tensor(float(10000000)), torch.tensor(float(10000000))]
    iter_stall = [0, 0]
    iter_num = 0
    max_iter = self.max_iter

    # While objective is still decreasing and iterations < max_iter
    while(max(iter_stall) < self.patience and iter_num < max_iter):
      if self.num_gpus > 1:
        o_funct = [torch.zeros(1,dtype=float).to(torch.device('cuda:0')), torch.zeros(1,dtype=float).to(torch.device('cuda:1'))]
      else:
        o_funct = [torch.zeros(1,dtype=float), torch.zeros(1,dtype=float)]
      o_funct_batch = [None, None]
      for batch_step in range(n_steps):
        Xbatch = [Xs[0][self.batch_size*batch_step:self.batch_size*(batch_step+1)],
                  Xs[1][self.batch_size*batch_step:self.batch_size*(batch_step+1)]]

        for vi in range(2):
          pre_view = (vi + 1) % 2
          partition_batch = partitions[pre_view][self.batch_size*batch_step:self.batch_size*(batch_step+1)]
          # Switch partitions and compute maximization
          partitions[vi][self.batch_size*batch_step:self.batch_size*(batch_step+1)], centroids[vi], o_funct_batch[vi] = self._em_step(Xbatch[vi], partition_batch, centroids[vi])

        o_funct[0] += o_funct_batch[0]
        o_funct[1] += o_funct_batch[1]
      # Track the number of iterations without improvement
      for view in range(2):
        if(objective[view] - o_funct[view] > self.tol * torch.abs(objective[view])):
            objective[view] = o_funct[view]
            iter_stall[view] = 0
        else:
            iter_stall[view] += 1
      iter_num += 1
    intertia = torch.sum(torch.tensor(objective,dtype=float))

    return intertia, centroids

  def _final_centroids(self, Xs, centroids):
    r''' Computes the final cluster centroids based on consensus samples across
        both views. Consensus samples are those that are assigned to the same
        partition in both views.

      Parameters
      ----------
      Xs : list of torch.Tensor
          - Xs length: n_views
          - Xs[i] shape: (n_samples, n_features_i)

      centroids : list of torch.Tensor
                - centroids length: n_views
                - centroids[i] shape: (n_clusters, n_features_i)
      '''
  
    # Compute consensus vectors for final clustering
    v1_consensus = list()
    v2_consensus = list()
    n_steps = int(Xs[0].shape[0]/self.batch_size)
    v1_partitions = list()
    v2_partitions = list()

    for batch_step in range(n_steps):

      v1_distances = self._compute_distance(Xs[0][self.batch_size*batch_step:self.batch_size*(batch_step+1)], centroids[0])
      v1_partitions.append(torch.argmin(v1_distances, dim=1).squeeze())
      v2_distances = self._compute_distance(Xs[1][self.batch_size*batch_step:self.batch_size*(batch_step+1)], centroids[1])
      v2_partitions.append(torch.argmin(v2_distances, dim=1).squeeze())

    v1_distances = self._compute_distance(Xs[0][self.batch_size*n_steps:Xs[0].shape[0]], centroids[0])
    v1_partitions.append(torch.argmin(v1_distances, dim=1).squeeze())
    v2_distances = self._compute_distance(Xs[1][self.batch_size*n_steps:Xs[1].shape[0]], centroids[1])
    v2_partitions.append(torch.argmin(v2_distances, dim=1).squeeze())
    v1_partitions = torch.cat(v1_partitions,dim=0)
    v2_partitions = torch.cat(v2_partitions,dim=0)
    
    for clust in range(self.n_clusters):
      # Find data points in the same partition in both views
      if self.num_gpus > 1:
        part_indices = (v1_partitions == clust) * (v2_partitions.to(torch.device('cuda:0')) == clust)
      else:
        part_indices = (v1_partitions == clust) * (v2_partitions == clust)

      # Recompute centroids based on these data points
      if (torch.sum(part_indices) != 0):
        cent1 = torch.mean(Xs[0][part_indices], dim=0)
        v1_consensus.append(cent1)
        cent2 = torch.mean(Xs[1][part_indices], dim=0)
        v2_consensus.append(cent2)

    # Check if there are no consensus vectors
    self.centroids = [None, None]
    if (len(v1_consensus) == 0):
      print('No distinct cluster centroids have been found.')
    else:
      self.centroids[0] = torch.stack(v1_consensus)
      self.centroids[1] = torch.stack(v2_consensus)
      self.n_clusters = self.centroids[0].shape[0]
  
  def check_views(self, Xs):
    r''' Checks the number of views
    Parameters
    ----------
    Xs : list of torch.Tensor
         - Xs length: n_views
         - Xs[i] shape: (n_samples, n_features_i)
    '''
    if len(Xs) != 2:
      raise ValueError('Number of views of data must be 2')

  def fit(self,Xs):
    r''' Performs clustering algorithm on multi-view data
    Parameters
    ----------
    Xs : list of torch.Tensor
         - Xs length: n_views
         - Xs[i] shape: (n_samples, n_features_i)
    
    '''
    self.check_views(Xs)
    if self.num_gpus > 1:
      Xs[0] = Xs[0].to(torch.device('cuda:0'))
      Xs[1] = Xs[1].to(torch.device('cuda:1'))
    else:
      Xs = [X.to(self.device) for X in Xs]
    if self.random_state is not None:
      torch.manual_seed(self.random_state)
    
    if self.n_jobs == -1:
      # Running algorithm only with one centroid seed
      inertia, centroids = self._one_init(Xs)
      self._final_centroids(Xs,centroids)
      self.labels_ = self.predict(Xs)
      return self
    else:
      #Ranning algorithm with n_init centroid seeds
      run_results = Parallel(n_jobs=self.n_jobs)(
      delayed(self._one_init)(Xs) for _ in range(self.n_init))
      intertias, centroids = zip(*run_results)
      max_ind = np.argmax(intertias)
      self._final_centroids(Xs,centroids[max_ind])
      self.labels_ = self.predict(Xs)
      return self

  def predict(self, Xs):
    r''' Predicts for the label for each input data points
    Parameters
    ----------
    Xs : list of torch.Tensor
         - Xs length: n_views
         - Xs[i] shape: (n_samples, n_features_i)
    '''
    dist1 = self._compute_distance(Xs[0], self.centroids[0])
    dist2 = self._compute_distance(Xs[1], self.centroids[1])
    dist_metric = dist1 + dist2
    labels = torch.argmin(dist_metric, dim=1).squeeze()
    return labels

