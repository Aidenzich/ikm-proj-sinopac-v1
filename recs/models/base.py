import os
import copy
import inspect
import pickle
from glob import glob
from datetime import datetime
import numpy as np
from ..utils.common import clip 

class Base:

    def __init__(self, name, trainable=True, verbose=False):
        """Build an base class of Neural Recommender Systems
        
        Parameters
        ----------
        name: string
            The name of the model
        
        trainable: bool
            When False, the model is not trained    
        
        """
        self.name = name
        self.trainable = trainable
        self.verbose = verbose
        self.train_set = None
        self.val_set = None        
        self.ignored_attrs = ["train_set", "val_set"]

    def reset_info(self):
        self.best_value = -np.Inf
        self.best_epoch = 0
        self.current_epoch = 0
        self.stopped_epoch = 0
        self.wait = 0

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k in self.ignored_attrs:
                continue
            setattr(result, k, copy.deepcopy(v))
        return result

    @classmethod
    def _get_init_params(cls):
        """Get initial parameters from the model constructor
        Parameters:
        ----------
        cls: Base class
        
        """
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            return []

        init_signature = inspect.signature(init)
        parameters = [p for p in init_signature.parameters.values() if p.name != "self"]

        return sorted([p.name for p in parameters])

    def clone(self, new_params=None):
        """Clone an instance of the model object.
        """
        new_params = {} if new_params is None else new_params
        init_params = {}
        for name in self._get_init_params():
            init_params[name] = new_params.get(name, copy.deepcopy(getattr(self, name)))

        return self.__class__(**init_params)

    def save(self, save_dir=None):
        """Save a recommender model to the filesystem.
        """
        if save_dir is None:
            return

        model_dir = os.path.join(save_dir, self.name)
        os.makedirs(model_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        model_file = os.path.join(model_dir, "{}.pkl".format(timestamp))

        saved_model = copy.deepcopy(self)

        pickle.dump(
            saved_model, open(model_file, "wb"), protocol=pickle.HIGHEST_PROTOCOL
        )

        if self.verbose:
            print("{} model is saved to {}".format(self.name, model_file))

        return model_file

    @staticmethod
    def load(model_path, trainable=False):
        """Load a recommender model from the filesystem.
        """
        if os.path.isdir(model_path):
            model_file = sorted(glob("{}/*.pkl".format(model_path)))[-1]
        else:
            model_file = model_path

        model = pickle.load(open(model_file, "rb"))
        model.trainable = trainable
        model.load_from = model_file  # for further loading

        return model

    def fit(self, train_set, val_set=None):
        self.reset_info()
        self.train_set = train_set.reset()
        self.val_set = None if val_set is None else val_set.reset()
        return self

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.
        """
        raise NotImplementedError("The algorithm is not able to make score prediction!")

    def rate(self, user_idx, item_idx, clipping=True):
        """Give a rating score between pair of user and item
        """
        try:
            rating_pred = self.score(user_idx, item_idx)
        except Exception:
            rating_pred = self.default_score()

        if clipping:
            rating_pred = clip(
                values=rating_pred,
                lower_bound=self.train_set.min_rating,
                upper_bound=self.train_set.max_rating,
            )

        return rating_pred

    def rank(self, user_idx, item_indices=None):
        """Rank all test items for a given user.
        """
        # obtain item scores from the model
        try:
            known_item_scores = self.score(user_idx)
        except Exception:
            known_item_scores = (
                np.ones(self.train_set.total_items) * self.default_score()
            )

        # check if the returned scores also cover unknown items
        # if not, all unknown items will be given the MIN score
        if len(known_item_scores) == self.train_set.total_items:
            all_item_scores = known_item_scores
        else:
            all_item_scores = np.ones(self.train_set.total_items) * np.min(
                known_item_scores
            )
            all_item_scores[: self.train_set.num_items] = known_item_scores

        # rank items based on their scores
        if item_indices is None:
            item_scores = all_item_scores[: self.train_set.num_items]
            item_rank = item_scores.argsort()[::-1]
        else:
            item_scores = all_item_scores[item_indices]
            item_rank = np.array(item_indices)[item_scores.argsort()[::-1]]

        return item_rank, item_scores