
import os
import copy
from tqdm.auto import trange
from ..base import Base
from ...utils import get_random_state


class NCFBase(Base):
    """Base class of NCF.    
    """
    def __init__(
        self,
        name="NCF",
        num_epochs=20,
        batch_size=256,
        num_neg=4,
        lr=0.001,
        learner="adam",        
        trainable=True,
        verbose=True,
        seed=None,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        import tensorflow.compat.v1 as tf
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.num_neg = num_neg
        self.lr = lr
        self.learner = learner    
        self.seed = seed
        self.random_state = get_random_state(seed)
        self.ignored_attrs.extend(
            [
                "graph",
                "user_id",
                "item_id",
                "labels",
                "interaction",
                "prediction",
                "loss",
                "train_op",
                "initializer",
                "saver",
                "sess",
            ]
        )

    def fit(self, train_set, val_set=None):
        """Fit the model.
        """
        Base.fit(self, train_set, val_set)

        if self.trainable:
            if not hasattr(self, "graph"):
                self.num_users = self.train_set.num_users
                self.num_items = self.train_set.num_items
                self._build_graph()
            self._fit_tf()

        return self

    def _build_graph(self):


        # less verbose TF
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        tf.logging.set_verbosity(tf.logging.ERROR)

        self.graph = tf.Graph()

    def _sess_init(self):
        

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        self.sess.run(self.initializer)

    def _step_update(self, batch_users, batch_items, batch_ratings):
        _, _loss = self.sess.run(
            [self.train_op, self.loss],
            feed_dict={
                self.user_id: batch_users,
                self.item_id: batch_items,
                self.labels: batch_ratings.reshape(-1, 1),
            },
        )
        return _loss

    def _fit_tf(self):
        loop = trange(self.num_epochs, disable=not self.verbose)
        for _ in loop:
            count = 0
            sum_loss = 0
            for i, (batch_users, batch_items, batch_ratings) in enumerate(
                self.train_set.uir_iter(
                    self.batch_size, shuffle=True, binary=True, num_zeros=self.num_neg
                )
            ):
                _loss = self._step_update(batch_users, batch_items, batch_ratings)
                count += len(batch_ratings)
                sum_loss += _loss * len(batch_ratings)
                if i % 10 == 0:
                    loop.set_postfix(loss=(sum_loss / count))

        loop.close()

    def save(self, save_dir=None):
        """Save a recommender model to the filesystem.
        """
        if save_dir is None:
            return

        model_file = Base.save(self, save_dir)
        # save TF weights
        self.saver.save(self.sess, model_file.replace(".pkl", ".cpt"))

        return model_file

    @staticmethod
    def load(model_path, trainable=False):
        """Load a recommender model from the filesystem.
        """
        model = Base.load(model_path, trainable)
        if hasattr(model, "pretrained"):  # NeuMF
            model.pretrained = False

        model._build_graph()
        model.saver.restore(model.sess, model.load_from.replace(".pkl", ".cpt"))

        return model
