"""
K-means clustering.
"""

import numpy as np
from matplotlib import pyplot as plt
import sys

def analyze_kmeans():
    """
    Top-level wrapper to iterate over a bunch of values of k and plot the
    distortions and misclassification rates.
    """
    X = np.genfromtxt("digit.txt")
    y = np.genfromtxt("labels.txt", dtype=int)
    distortions = []
    errs = []
    ks = range(1, 11) 
    for k in ks:
        distortion, err = analyze_one_k(X, y, k)
        distortions.append(distortion)
        errs.append(err)
    fig, ax = plt.subplots(2, figsize=(8, 6))
    ax[0].plot(ks, distortions, marker=".")
    ax[0].set_ylabel("Distortion")
    ax[1].plot(ks, errs, marker=".")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("Mistake rate")
    ax[0].set_title("k-means performance")
    fig.savefig("kmeans.png")


def analyze_one_k(X, y, k):
    """
    Run the k-means analysis for a single value of k. Return the distortion and
    the mistake rate.
    """
    print "Running k-means with k={0}".format(k)
    clust = cluster(X, y, k)
    print "Computing classification error."
    err = compute_mistake_rate(y, clust)
    return clust["distortion"], err


def cluster(X, y, k, n_starts=5):
    """
    Run k-means a total of n_starts times. Returns the results from the run that
    had the lowest within-group sum of squares (i.e. the lowest distortion).

    Inputs
    ------
    X is an NxD matrix of inputs.
    y is a Dx1 vector of labels.
    n_starts says how many times to randomly re-initialize k-means. You don't
        need to change this.

    Outputs
    -------
    The output is a dictionary with the following fields:
    Mu is a kxD matrix of cluster centroids
    z is an Nx1 vector assigning points to clusters. So, for instance, if z[4] =
        2, then the algorithm has assigned the 4th data point to the second
        cluster.
    distortion is the within-group sum of squares, a number.
    """
    def loop(X, i):
        """
        A single run of clustering.
        """
        Mu = initialize(X, k)
        N = X.shape[0]
        z = np.repeat(-1, N)        # So that initially all assignments change.
        while True:
            old_z = z
            z = assign(X, Mu)       # The vector of assignments z.
            Mu = update(X, z, k)    # Update the centroids
            if np.all(z == old_z):
                distortion = compute_distortion(X, Mu, z)
                return dict(Mu=Mu, z=z, distortion=distortion)

    # Main function body
    print "Performing clustering."
    results = [loop(X, i) for i in range(n_starts)]
    best = min(results, key=lambda entry: entry["distortion"])
    best["digits"] = label_clusters(y, k, best["z"])
    return best


def assign(X, Mu):
    """
    Assign each entry to the closest centroid. Return an Nx1 vector of
    assignments z.
    X is the NxD matrix of inputs.
    Mu is the kxD matrix of cluster centroids.
    """
    mu_rows = len(Mu[:,0])
    X_rows = len(X[:,0])
    z = []
    for i in range(0,X_rows):
        row = np.matrix([X[i,:]] * mu_rows)
        minimum = np.argmin(np.square(np.linalg.norm(np.subtract(Mu,row),axis=1)))
        z.append(minimum)
    return z


def update(X, z, k):
    """
    Update the cluster centroids given the new assignments. Return a kxD matrix
    of cluster centroids Mu.
    X is the NxD inputs as always.
    z is the Nx1 vector of cluster assignments. -- gives index for row i which cluster it belongs too
    k is the number of clusters.
    """
    ## z = [1,2,3,...k,4,3,5,k-1]
    N = len(X[:,0])
    D = len(X[0,:])
    Mu = np.zeros(shape = (k,D))
    points_per_cluster = dict.fromkeys(range(k),0)
    for i in range(0,N):
        # get row i
        row = X[i,:] * 1.0
        # ith value at z dictates what cluster it is in and which row to add it to for final average
        cluster = z[i]
        Mu[cluster,:] = Mu[cluster,:] + row
        points_per_cluster[cluster] = points_per_cluster[cluster] + 1
    #Now take the average each cluster
    np.seterr(divide='ignore', invalid='ignore')
    for cluster in range(k):
        Mu[cluster,:] = Mu[cluster,:] / np.array(points_per_cluster[cluster], dtype=np.float128)

    return Mu


def compute_distortion(X, Mu, z):
    """
    Compute the distortion (i.e. within-group sum of squares) implied by NxD
    data X, kxD centroids Mu, and Nx1 assignments z.
    """
    N = len(X[:,0])
    distortion = 0
    for i in range(0,N):
        row = X[i,:]
        cluster = z[i]
        distortion+=np.square(np.linalg.norm(np.subtract(Mu[cluster,:],row))) 
    return distortion


def initialize(X, k):
    """
    Randomly initialize the kxD matrix of cluster centroids Mu. Do this by
    choosing k data points randomly from the data set X.
    """
    subset = np.random.randint(len(X[:,0]),size=k)
    Mu = X[subset,:]
    return Mu


def label_clusters(y, k, z):
    """
    Label each cluster with the digit that occurs most frequently for points
    assigned to that cluster.
    Return a kx1 vector labels with the label for each cluster.
    For instance: if 20 points assigned to cluster 0 have label "3", and 40 have
    label "5", then labels[0] should be 5.

    y is the Nx1 vector of digit labels for the data X
    k is the number of clusters
    z is the Nx1 vector of cluster assignments.
    """

    labels = [0] * k
    ## for each cluster we need to know which the number of points assigned to it
    ## then we need to know what was assigned to that point
    cluster_labels = {j: dict.fromkeys(range(k),0) for j in range(k)}
    N = len(y)
    for i in range(N):
        cluster = z[i]
        label = y[i]
        if label not in cluster_labels[cluster]:
            cluster_labels[cluster][label] = 0
        cluster_labels[cluster][label] = cluster_labels[cluster][label] + 1

    for i,key in enumerate(cluster_labels):
        #each key in the dictionary is the label and the value is the number of points for it
        # for each cluster there is dict of label-value pairs, we choose the label with most value
        label = max(cluster_labels[key], key=cluster_labels[key].get)
        labels[i] = label
    
    return np.array(labels)


def compute_mistake_rate(y, clust):
    """
    Compute the mistake rate as discussed in section 3.4 of the homework.
    y is the Nx1 vector of true labels.
    clust is the output of a run of clustering. Two fields are relevant:
    "digits" is a kx1 vector giving the majority label for each cluster
    "z" is an Nx1 vector of final cluster assignments.
    """
    def zero_one_loss(xs, ys):
        return sum(xs != ys) / float(len(xs))

    y_hat = clust["digits"][clust["z"]]
    return zero_one_loss(y, y_hat)


def main():
    analyze_kmeans()


if __name__ == '__main__':
    main()
