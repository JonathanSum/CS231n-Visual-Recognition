import numpy as np

N, D = 3, 4
X = np.random.randon(N, D)

# centered the data
X -= np.mean(X, axis=0)  # No axis is fine too

# resizing the data that has the same size of up down and right left

X /= np.std(X,axis=0)  # Another form of this preprocessing normalizes each dimension so that the min and max along the dimension is -1 and 1 respectively. It only makes sense to apply this preprocessing if you have a reason to believe that different input features have different scales (or units.

# PCA and whitening
# NOTE: it can reduce the dimensionality.
# After this operation, we would have reduced the original dataset of size [N x D] to one of size [N x 100], keeping
#  the 100 dimensions of the data that contain the most variance. It is very often the case that you can get very good
#  performance by training linear classifiers or neural networks on the PCA-reduced datasets, obtaining savings in
#  both space and time.
X -= np.mean(X, axis=0)
cov = np.dot(X.T, X) / X.shape[0]
U, S, V = np.linalg.svd(cov)
Xrot = np.dot(X, U)  # decorrelate the data
Xrot_reduced = np.dot(X, U[:,:100]) # Xrot_reduced becomes [Nx 100]









