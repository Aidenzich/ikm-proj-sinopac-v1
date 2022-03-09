import numpy as np
import torch
from ..base import Base

class VAECF(Base):
    """Variational Autoencoder for Collaborative Filtering.
    Liang, Dawen, Rahul G. Krishnan, Matthew D. Hoffman, and Tony Jebara. "Variational autoencoders for collaborative filtering."
    """

    def __init__(
        self,
        name="VAECF",
        k=10,
        autoencoder_structure=[20],
        act_fn="tanh",
        likelihood="mult",
        n_epochs=100,
        batch_size=100,
        learning_rate=0.001,
        beta=1.0,
        trainable=True,
        verbose=False,
        seed=None,
        use_gpu=False,
    ):
        Base.__init__(self, name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.autoencoder_structure = autoencoder_structure
        self.act_fn = act_fn
        self.likelihood = likelihood
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.beta = beta
        self.seed = seed
        self.use_gpu = use_gpu

    def fit(self, train_set, val_set=None):
        """Fit the model to observations.

        Parameters
        ----------
        train_set: :obj:`cornac.data.Dataset`, required
            User-Item preference data as well as additional modalities.

        val_set: :obj:`cornac.data.Dataset`, optional, default: None
            User-Item preference data for model selection purposes (e.g., early stopping).

        Returns
        -------
        self : object
        """
        Base.fit(self, train_set, val_set)

        
        from .vaecf import VAE, learn

        self.device = (
            torch.device("cuda:0")
            if (self.use_gpu and torch.cuda.is_available())
            else torch.device("cpu")
        )

        if self.trainable:
            if self.seed is not None:
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed(self.seed)

            if not hasattr(self, "vae"):
                data_dim = train_set.matrix.shape[1]
                self.vae = VAE(
                    self.k,
                    [data_dim] + self.autoencoder_structure,
                    self.act_fn,
                    self.likelihood,
                ).to(self.device)

            learn(
                self.vae,
                self.train_set,
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                learn_rate=self.learning_rate,
                beta=self.beta,
                verbose=self.verbose,
                device=self.device,
            )

        elif self.verbose:
            print("%s is trained already (trainable = False)" % (self.name))

        return self

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.

        Parameters
        ----------
        user_idx: int, required
            The index of the user for whom to perform score prediction.

        item_idx: int, optional, default: None
            The index of the item for which to perform score prediction.
            If None, scores for all known items will be returned.

        Returns
        -------
        res : A scalar or a Numpy array
            Relative scores that the user gives to the item or to all known items

        """
        import torch

        if item_idx is None:
            if self.train_set.is_unknown_user(user_idx):
                raise Exception(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )

            x_u = self.train_set.matrix[user_idx].copy()
            x_u.data = np.ones(len(x_u.data))
            z_u, _ = self.vae.encode(
                torch.tensor(x_u.A, dtype=torch.float32, device=self.device)
            )
            known_item_scores = self.vae.decode(z_u).data.cpu().numpy().flatten()

            return known_item_scores
        else:
            if self.train_set.is_unknown_user(user_idx) or self.train_set.is_unknown_item(
                item_idx
            ):
                raise Exception(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            x_u = self.train_set.matrix[user_idx].copy()
            x_u.data = np.ones(len(x_u.data))
            z_u, _ = self.vae.encode(
                torch.tensor(x_u.A, dtype=torch.float32, device=self.device)
            )
            user_pred = (
                self.vae.decode(z_u).data.cpu().numpy().flatten()[item_idx]
            )  # Fix me I am not efficient

            return user_pred