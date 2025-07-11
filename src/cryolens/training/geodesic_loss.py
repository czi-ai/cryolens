import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MetricTensor(nn.Module):
    """Learnable Riemannian Metric tensor

    Learns the local geometry of the latent space. It outputs a positive 
    definite matrix (the metric tensor) for any point z. To ensure the 
    matrix is positive definite, output a lower triangular matrix L and 
    compute G = LL^T.
    """
    def __init__(self, latent_dim: int, *, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.tril_elems = latent_dim * (latent_dim + 1) // 2

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.tril_elems)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]

        # get the elements of the lower triangular matrix
        tril_flat = self.net(z).view(batch_size, -1).contiguous()

        # create the lower triangular matrix, L
        # Ensure L has the same dtype as the input
        L = torch.zeros(batch_size, self.latent_dim, self.latent_dim, 
                       device=z.device, dtype=z.dtype)
        tril_indices = torch.tril_indices(
            row=self.latent_dim, 
            col=self.latent_dim, 
            offset=0, 
            device=z.device
        )
        tril_flat = tril_flat.view(batch_size, self.tril_elems)
        L[:, tril_indices[0], tril_indices[1]] = tril_flat

        # ensure the diagonal is positive for stability
        L[:, torch.arange(self.latent_dim), torch.arange(self.latent_dim)] = F.softplus(
            L[:, torch.arange(self.latent_dim), torch.arange(self.latent_dim)]
        )

        # the metric tensor G = LL^T, which is guaranteed to be positive definite
        # G shape: (batch_size, latent_dim, latent_dim)
        G = torch.bmm(L, L.transpose(1, 2))
        return G

    def geodesic_distance_approximation(
        self,
        z1: torch.Tensor, 
        z2: torch.Tensor, 
        *, 
        steps: int = 10
    ) -> torch.Tensor:
        """
        Approximates the geodesic distance by discretizing the path between z1 and z2.
        Assumption is that the path is locally smooth.
        """
        # create a series of points interpolating between z1 and z2
        t = torch.linspace(0, 1, steps, device=z1.device).view(-1, 1)
        path = z1 * (1 - t) + z2 * t  # (steps, latent_dim)

        # tangent vectors are the differences between consecutive points
        tangent_vectors = path[1:] - path[:-1] # (steps-1, latent_dim)

        # metric at the midpoint of each segment for better accuracy
        mid_points = (path[1:] + path[:-1]) / 2
        
        # calculate the metric tensor G at each midpoint
        # (steps-1, latent_dim, latent_dim)
        G = self.forward(mid_points)

        # calculate the squared length of each segment: v^T * G * v
        v = tangent_vectors.unsqueeze(1) # (steps-1, 1, latent_dim)
        v_t = tangent_vectors.unsqueeze(2) # (steps-1, latent_dim, 1) 
        segment_lengths_sq = torch.bmm(torch.bmm(v, G), v_t).squeeze()
        
        # total distance is the sum of the lengths of the segments
        total_distance = torch.sum(torch.sqrt(segment_lengths_sq + 1e-8))
        
        return total_distance


class GeodesicAffinityLoss(torch.nn.Module):
    """Affinity loss based on pre-calculated shape similarity. 

    Uses a geodesic path length as a measure of latent similarity.

    Parameters
    ----------
    lookup : np.ndarray (M, M)
        A square symmetric matrix where each column and row is the index of an
        object from the training set, consisting of M different objects. The
        value at (i, j) is a scalar value encoding the shape similarity between
        objects i and j, pre-calculated using some shape (or other) metric. The
        identity of the matrix should be 1 since these objects are the same
        shape. The affinity similarity should be normalized to the range
        (-1, 1).

    Notes
    -----
    The final loss is calculated using L1-norm. This could be changed, e.g.
    L2-norm. Not sure what the best one is yet.
    """

    def __init__(self, lookup: torch.Tensor, latent_dim: int, device: torch.device):
        super().__init__()
        self.device = device
        # Register lookup as a buffer so it's saved with the model
        self.register_buffer('lookup', torch.tensor(lookup).to(device))
        # Register MetricTensor as a submodule so it's saved with checkpoints
        self.metric = MetricTensor(latent_dim=latent_dim)
        # Move metric to device after initialization
        self.metric = self.metric.to(device)
        self.l1loss = torch.nn.L1Loss()

    def forward(
        self, y_true: torch.Tensor, y_pred: torch.Tensor
    ) -> torch.Tensor:
        """Return the affinity loss.

        Parameters
        ----------
        y_true : torch.Tensor (N, )
            A vector of N objects in the mini-batch of the indices representing
            the identity of the object as an index. These indices should
            correspond to the rows and columns of the `lookup` table.
        y_pred : torch.Tensor (N, latent_dims)
            An array of latent encodings of the N objects.

        Returns
        -------
        loss : torch.Tensor
            The affinity loss.
        """

        affinity_loss = 0.0
        batch_size = y_pred.shape[0]

        # Ensure consistent dtype (use float32 for loss calculation)
        y_pred = y_pred.float()

        # calculate the upper triangle only
        for i in range(batch_size):
            for j in range(i + 1, batch_size):

                # get the pre-calculated affinity score from the matrix
                label_i, label_j = y_true[i].long(), y_true[j].long()
                prior_affinity = self.lookup[label_i, label_j]
                
                # get the learned distance from the model. use the mean 'mu' for 
                # calculating distance to have a stable representation
                dist = self.metric.geodesic_distance_approximation(
                    y_pred[i], 
                    y_pred[j]
                )
                
                # use a negative exponential to map distance to similarity (0 to 1)
                learned_affinity = torch.exp(-dist) 
                
                # L1 norm of the difference
                affinity_loss += torch.abs(prior_affinity - learned_affinity)

        return torch.mean(affinity_loss)