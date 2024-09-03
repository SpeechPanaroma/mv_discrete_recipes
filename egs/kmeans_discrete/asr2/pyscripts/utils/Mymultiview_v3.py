import numpy as np
import torch
import logging
from joblib import delayed,Parallel
import sys
import time
import GPUtil
class MyMultiviewKMeansCluster():
  def __init__(self,n_clusters, max_iter, patience, tol,init,n_jobs,n_init,device1,device2):
    self.n_clusters = n_clusters
    self.max_iter = max_iter
    self.patience = patience
    self.tol = tol
    self.centroids = list()
    self.init = init
    self.n_jobs = n_jobs
    self.n_init = n_init
    self.device1 = device1
    self.device2 = device2

  def _compute_distance(self, X, centroids):
 
    # X = X.float().to(self.device3)
    # centroids= centroids.float().to(self.device3)
    # dist = torch.cdist(X,centroids).to(current_device)
    return torch.cdist(X.float(),centroids.float())

#   def _compute_distance(self, X, centroids):
#     start = time.time()
#     dist = torch.cdist(X.float(),centroids.float())
#     logger.info("dist clculation time: "+ str(time.time()-start))
#     return dist

  def _init_centroids(self,Xs):
    logging.info("entered in init centroids")
    if(self.init == "random"):
      indices = np.random.choice(Xs[0].shape[0], self.n_clusters).tolist()
      centers1 = Xs[0][indices]
      centers2 = Xs[1][indices]
      centroids = [centers1, centers2]
    else:
      indices = list()
      centers2 = list()
      indices.append(int(torch.randint(0,Xs[1].shape[0],(1,))))
      centers2.append(Xs[1][indices[0], :])

      # Compute the remaining n_cluster centroids
      for cent in range(self.n_clusters - 1):
        dists = self._compute_distance(torch.stack(centers2),Xs[1])
        dists = torch.min(dists, dim=1)
        max_index = torch.argmax(dists.values)
        indices.append(int(max_index))
        centers2.append(Xs[1][max_index])

      centers1 = Xs[0][indices].to(self.device1)
      centers2 = torch.stack(centers2).to(self.device2)
      centroids = [centers1, centers2]
    return centroids

  def _em_step(self, X, partition, centroids):
    n_samples = X.shape[0]
    new_centers = list()
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
    # Initialize centroids for clustering
    centroids = self._init_centroids(Xs)

    # Initializing partitions, objective value, and loop vars
    distances = self._compute_distance(Xs[1], centroids[1])
    parts = torch.argmin(distances, dim=1).squeeze()
    partitions = [None, parts]
    objective = [torch.tensor(float(10000000)).to(self.device1), torch.tensor(float(10000000)).to(self.device2)]
    o_funct = [None, None]
    iter_stall = [0, 0]
    iter_num = 0
    max_iter = self.max_iter

    # While objective is still decreasing and iterations < max_iter
    while(max(iter_stall) < self.patience and iter_num < max_iter):
        logging.info("Number of iterations: "+str(iter_num))
        for vi in range(2):
            pre_view = (iter_num + 1) % 2
            # Switch partitions and compute maximization
            partitions[vi], centroids[vi], o_funct[vi] = self._em_step(
                Xs[vi], partitions[pre_view], centroids[vi])
        iter_num += 1
        # Track the number of iterations without improvement
        for view in range(2):
          if(objective[view] - o_funct[view] > self.tol * torch.abs(objective[view])):
              objective[view] = o_funct[view]
              iter_stall[view] = 0
          else:
              iter_stall[view] += 1    
    intertia = torch.sum(torch.tensor(objective,dtype=float))

    return intertia, centroids

  def _final_centroids(self, Xs, centroids):
    # Compute consensus vectors for final clustering
    v1_consensus = list()
    v2_consensus = list()
    v1_distances = self._compute_distance(Xs[0], centroids[0])
    v1_partitions = torch.argmin(v1_distances, dim=1).squeeze()
    v2_distances = self._compute_distance(Xs[1], centroids[1])
    v2_partitions = torch.argmin(v2_distances, dim=1).squeeze()
    for clust in range(self.n_clusters):
      # Find data points in the same partition in both views
      part_indices = (v1_partitions == clust) * (v2_partitions.to(self.device1) == clust)

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

  def print_gpu_usage(self):
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU ID {gpu.id}: {gpu.memoryUsed} MB / {gpu.memoryTotal} MB ({gpu.memoryUtil * 100}%)")
  
  def fit(self,Xs):
    Xs[0] = Xs[0].to(self.device1)
    Xs[1] = Xs[1].to(self.device2)
    # inertia, centroids = self._one_init(Xs)
    # self._final_centroids(Xs,centroids)
    run_results = Parallel(n_jobs=self.n_jobs)(
    delayed(self._one_init)(Xs) for _ in range(self.n_init))
    intertias, centroids = zip(*run_results)
    max_ind = np.argmax(intertias)
    self._final_centroids(Xs,centroids[max_ind])
    self.print_gpu_usage()

  def predict(self, Xs):
    dist1 = self._compute_distance(Xs[0], self.centroids[0])
    dist2 = self._compute_distance(Xs[1], self.centroids[1])
    dist_metric = dist1 + dist2
    labels = torch.argmin(dist_metric, dim=1).squeeze()
    return labels

class MultiviewKMeansClusterMiniBatch():
  def __init__(self, n_clusters, max_iter, patience, tol, init, batch_size,n_jobs,n_init,device1,device2):
    self.n_clusters = n_clusters
    self.max_iter = max_iter
    self.patience = patience
    self.tol = tol
    self.centroids = list()
    self.init = init
    self.batch_size = batch_size
    self.n_jobs = n_jobs
    self.n_init = n_init
    self.device1 = device1
    self.device2 = device2

  def _compute_distance(self, X, centroids):
    return torch.cdist(X.float(),centroids.float())

  def _init_centroids(self,Xs):
    logging.info("View 0 is on device GPU: "+str(Xs[0].get_device()))
    logging.info("View 1 is on device GPU: "+str(Xs[1].get_device()))
    logging.info("batch_size:" + str(self.batch_size))
    if(self.init == "random"):
      indices = np.random.choice(Xs[0].shape[0], self.n_clusters).tolist()
      centers1 = Xs[0][indices]
      centers2 = Xs[1][indices]
      centroids = [centers1, centers2]
    else:
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
          dists.append(torch.min(d, dim=1).values)
        dists = torch.stack(dists)
        max_index = torch.argmax(dists)
        indices.append(int(max_index))
        centers2.append(Xs[1][max_index])

      centers1 = Xs[0][indices]
      centers2 = torch.stack(centers2)
      centroids = [centers1, centers2]
    return centroids

  def _em_step(self, X, partition, centroids, batch_step):
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
    partitions = [torch.zeros(parts.size()).to(self.device1), parts]
    objective = [torch.tensor(float(10000000000)).to(self.device1), torch.tensor(float(10000000000)).to(self.device2)]    
    iter_stall = [0, 0]
    iter_num = 0
    max_iter = self.max_iter

    # While objective is still decreasing and iterations < max_iter
    while(max(iter_stall) < self.patience and iter_num < max_iter):
      o_funct = [torch.zeros(1,dtype=float).to(self.device1), torch.zeros(1,dtype=float).to(self.device2)]
      o_funct_batch = [None, None]
      for batch_step in range(n_steps):
        # logging.info('Batch: '+str(batch_step)+"/Iteration: "+str(iter_num))
        Xbatch = [Xs[0][self.batch_size*batch_step:self.batch_size*(batch_step+1)],
                  Xs[1][self.batch_size*batch_step:self.batch_size*(batch_step+1)]]

        for vi in range(2):
          pre_view = (vi + 1) % 2 
          partition_batch = partitions[pre_view][self.batch_size*batch_step:self.batch_size*(batch_step+1)]
          # Switch partitions and compute maximization
          partitions[vi][self.batch_size*batch_step:self.batch_size*(batch_step+1)], centroids[vi], o_funct_batch[vi] = self._em_step(Xbatch[vi], partition_batch, centroids[vi], batch_step)
        o_funct[0] += o_funct_batch[0]
        o_funct[1] += o_funct_batch[1]
      # Track the number of iterations without improvement
      for view in range(2):
        if(objective[view] - o_funct[view] > self.tol * torch.abs(objective[view])):
            objective[view] = o_funct[view]
            iter_stall[view] = 0
        else:
            iter_stall[view] += 1 
      logging.info('Objective: '+str(objective)+"/Iteration: "+str(iter_num))
      iter_num += 1
    intertia = torch.sum(torch.tensor(objective,dtype=float))

    return intertia, centroids

  def _final_centroids(self, Xs, centroids):
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
      part_indices = (v1_partitions == clust) * (v2_partitions.to(self.device1) == clust)

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

  def fit(self,Xs):
    logging.info("Entered in fit")
    Xs[0] = Xs[0].to(self.device1)
    Xs[1] = Xs[1].to(self.device2)
    logging.info("Xs size: "+str(Xs[0].element_size() * Xs[0].nelement()*2))
    logging.info("features loaded on both gpu")
    # inertia, centroids = self._one_init(Xs)
    # self._final_centroids(Xs,centroids)
    run_results = Parallel(n_jobs=self.n_jobs,prefer="threads")(
    delayed(self._one_init)(Xs) for _ in range(self.n_init))
    intertias, centroids = zip(*run_results)
    max_ind = np.argmax(intertias)
    self._final_centroids(Xs,centroids[max_ind])
    # centroids_list = ['init_centroids1709974582.9024014.pt',
    #                   'init_centroids1709974817.2736454.pt',
    #                   'init_centroids1709974866.2201543.pt',
    #                   'centroids.pt']
    # inertia = list()
    # centroids = list()
    # for i in centroids_list:
    #   init_centroid = torch.load(i)   
    #   iner, cent = self._one_init(Xs,init_centroid)
    #   centroids.append(cent)
    #   inertia.append(iner)
    # max_ind = np.argmax(inertia)
    # print(max_ind)
    # self._final_centroids(Xs,centroids[max_ind])
    # inertia, centroids = self._one_init(Xs)
    # self._final_centroids(Xs,centroids)
    # return inertia

  def predict(self, Xs):
    dist1 = self._compute_distance(Xs[0], self.centroids[0])
    dist2 = self._compute_distance(Xs[1], self.centroids[1])
    dist_metric = dist1 + dist2
    labels = torch.argmin(dist_metric, dim=1).squeeze()
    return labels
