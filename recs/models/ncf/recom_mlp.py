
import numpy as np
from .recom_ncf_base import NCFBase


class MLP(NCFBase):
    """Multi-Layer Perceptron.
    * He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017, April). Neural collaborative filtering. \
    In Proceedings of the 26th international conference on world wide web (pp. 173-182).
    """

    def __init__(
        self,
        name="MLP",
        layers=(64, 32, 16, 8),
        act_fn="relu",
        reg_layers=(0.0, 0.0, 0.0, 0.0),
        num_epochs=20,
        batch_size=256,
        num_neg=4,
        lr=0.001,
        learner="adam",        
        trainable=True,
        verbose=True,
        seed=None,
    ):
        super().__init__(
            name=name,
            trainable=trainable,
            verbose=verbose,
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_neg=num_neg,
            lr=lr,
            learner=learner,            
            seed=seed,
        )
        import tensorflow.compat.v1 as tf
        self.layers = layers
        self.act_fn = act_fn
        self.reg_layers = reg_layers

    def _build_graph(self):

        from .ops import mlp, loss_fn, train_fn

        super()._build_graph()
        with self.graph.as_default():
            tf.set_random_seed(self.seed)

            self.user_id = tf.placeholder(shape=[None], dtype=tf.int32, name="user_id")
            self.item_id = tf.placeholder(shape=[None], dtype=tf.int32, name="item_id")
            self.labels = tf.placeholder(
                shape=[None, 1], dtype=tf.float32, name="labels"
            )

            self.interaction = mlp(
                uid=self.user_id,
                iid=self.item_id,
                num_users=self.num_users,
                num_items=self.num_items,
                layers=self.layers,
                reg_layers=self.reg_layers,
                act_fn=self.act_fn,
                seed=self.seed,
            )
            logits = tf.layers.dense(
                self.interaction,
                units=1,
                name="logits",
                kernel_initializer=tf.initializers.lecun_uniform(self.seed),
            )
            self.prediction = tf.nn.sigmoid(logits)

            self.loss = loss_fn(labels=self.labels, logits=logits)
            self.train_op = train_fn(
                self.loss, learning_rate=self.lr, learner=self.learner
            )

            self.initializer = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

        self._sess_init()

    def score(self, user_idx, item_idx=None):
        """Predict the scores/ratings of a user for an item.
        """
        if item_idx is None:
            if self.train_set.is_unknown_user(user_idx):
                raise Exception(
                    "Can't make score prediction for (user_id=%d)" % user_idx
                )

            known_item_scores = self.sess.run(
                self.prediction,
                feed_dict={
                    self.user_id: np.ones(self.train_set.num_items) * user_idx,
                    self.item_id: np.arange(self.train_set.num_items),
                },
            )
            return known_item_scores.ravel()
        else:
            if self.train_set.is_unknown_user(user_idx) or self.train_set.is_unknown_item(
                item_idx
            ):
                raise Exception(
                    "Can't make score prediction for (user_id=%d, item_id=%d)"
                    % (user_idx, item_idx)
                )

            user_pred = self.sess.run(
                self.prediction,
                feed_dict={self.user_id: [user_idx], self.item_id: [item_idx]},
            )
            return user_pred.ravel()
