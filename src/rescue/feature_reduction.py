import torch


class TorchIncrementalPCA:
    """
    Incremental PCA in PyTorch without mean centering.
    Designed for unit-normalized features (e.g. LSeg, CLIP dense features)
    where centering distorts the embedding space.
    Processes one batch at a time using SVD-merge approach.
    """
    def __init__(self, n_components=10, device=None, components=None, singular_values=None, n_samples_seen=0):
        if components is not None:
            assert n_components == components.shape[0], "n_components must match the number of columns in components"
            assert singular_values is not None, "singular_values must be provided"
            assert n_components == singular_values.shape[0], "n_components must match the number of rows in singular_values"
            self.components = components
            self.singular_values = singular_values
            self.n_samples_seen = n_samples_seen
        else:
            self.components = None
            self.singular_values = None
            self.n_samples_seen = 0

        self.n_components = n_components
        self.device = device if device is not None else 'cuda' if torch.cuda.is_available() else 'cpu'

    def partial_fit(self, X):
        """X: (n_samples, n_features)"""
        X = X.to(self.device).float()
        self.n_samples_seen += X.shape[0]

        if self.components is None:
            _, S, Vh = torch.linalg.svd(X, full_matrices=False)
            self.components = Vh[:self.n_components]
            self.singular_values = S[:self.n_components]
        else:
            combined = torch.cat([
                self.components * self.singular_values.unsqueeze(1),
                X
            ], dim=0)
            _, S, Vh = torch.linalg.svd(combined, full_matrices=False)
            self.components = Vh[:self.n_components]
            self.singular_values = S[:self.n_components]

    def transform(self, X):
        """X: (n_samples, n_features) -> (n_samples, n_components)
        Works for both image patches and text embeddings.
        """
        return X.to(self.device).float() @ self.components.T

    def inverse_transform(self, X_reduced):
        """X_reduced: (n_samples, n_components) -> (n_samples, n_features)"""
        return X_reduced @ self.components

    @property
    def explained_variance_ratio_(self):
        total = (self.singular_values ** 2).sum()
        return (self.singular_values ** 2) / total

    def save(self, path):
        torch.save({
            'components': self.components,
            'singular_values': self.singular_values,
            'n_samples_seen': self.n_samples_seen,
            'n_components': self.n_components,
        }, path)

    @classmethod
    def load(cls, path, device=None):
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(path, map_location=device)
        model = cls(n_components=checkpoint['n_components'], device=device)
        model.components = checkpoint['components']
        model.singular_values = checkpoint['singular_values']
        model.n_samples_seen = checkpoint['n_samples_seen']
        return model