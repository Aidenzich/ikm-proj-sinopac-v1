
import numpy as np
from .recom_ncf_base import NCFBase


class NeuMF(NCFBase):
    """Neural Matrix Factorization.
    """
    def __init__(
        self,
        name="NeuMF",
        num_factors=8,
        layers=(64, 32, 16, 8),
        act_fn="relu",
        reg_mf=0.0,
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
        self.num_factors = num_factors
        self.layers = layers
        self.act_fn = act_fn
        self.reg_mf = reg_mf
        self.reg_layers = reg_layers
        self.pretrained = False
        self.ignored_attrs.extend(
            [
                "gmf_user_id",
                "mlp_user_id",
                "gmf_model",
                "mlp_model",
                "alpha",
            ]
        )

    def pretrain(self, gmf_model, mlp_model, alpha=0.5):
        """Provide pre-trained GMF and MLP models. Section 3.4.1 of the paper.
        """
        self.pretrained = True
        self.gmf_model = gmf_model
        self.mlp_model = mlp_model
        self.alpha = alpha
        return self

    def _build_graph(self):
        
        from .ops import gmf, mlp, loss_fn, train_fn

        super()._build_graph()
        with self.graph.as_default():
            tf.set_random_seed(self.seed)

            self.gmf_user_id = tf.placeholder(
                shape=[None], dtype=tf.int32, name="gmf_user_id"
            )
            self.mlp_user_id = tf.placeholder(
                shape=[None], dtype=tf.int32, name="mlp_user_id"
            )
            self.item_id = tf.placeholder(shape=[None], dtype=tf.int32, name="item_id")
            self.labels = tf.placeholder(
                shape=[None, 1], dtype=tf.float32, name="labels"
            )

            gmf_feat = gmf(
                uid=self.gmf_user_id,
                iid=self.item_id,
                num_users=self.num_users,
                num_items=self.num_items,
                emb_size=self.num_factors,
                reg_user=self.reg_mf,
                reg_item=self.reg_mf,
                seed=self.seed,
            )
            mlp_feat = mlp(
                uid=self.mlp_user_id,
                iid=self.item_id,
                num_users=self.num_users,
                num_items=self.num_items,
                layers=self.layers,
                reg_layers=self.reg_layers,
                act_fn=self.act_fn,
                seed=self.seed,
            )

            self.interaction = tf.concat([gmf_feat, mlp_feat], axis=-1)
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

        if self.pretrained:
            gmf_kernel = self.gmf_model.sess.run(
                self.gmf_model.sess.graph.get_tensor_by_name("logits/kernel:0")
            )
            gmf_bias = self.gmf_model.sess.run(
                self.gmf_model.sess.graph.get_tensor_by_name("logits/bias:0")
            )
            mlp_kernel = self.mlp_model.sess.run(
                self.mlp_model.sess.graph.get_tensor_by_name("logits/kernel:0")
            )
            mlp_bias = self.mlp_model.sess.run(
                self.mlp_model.sess.graph.get_tensor_by_name("logits/bias:0")
            )
            logits_kernel = np.concatenate(
                [self.alpha * gmf_kernel, (1 - self.alpha) * mlp_kernel]
            )
            logits_bias = self.alpha * gmf_bias + (1 - self.alpha) * mlp_bias

            for v in self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                if v.name.startswith("GMF"):
                    sess = self.gmf_model.sess
                    self.sess.run(
                        tf.assign(v, sess.run(sess.graph.get_tensor_by_name(v.name)))
                    )
                elif v.name.startswith("MLP"):
                    sess = self.mlp_model.sess
                    self.sess.run(
                        tf.assign(v, sess.run(sess.graph.get_tensor_by_name(v.name)))
                    )
                elif v.name.startswith("logits/kernel"):
                    self.sess.run(tf.assign(v, logits_kernel))
                elif v.name.startswith("logits/bias"):
                    self.sess.run(tf.assign(v, logits_bias))

    def _step_update(self, batch_users, batch_items, batch_ratings):
        _, _loss = self.sess.run(
            [self.train_op, self.loss],
            feed_dict={
                self.gmf_user_id: batch_users,
                self.mlp_user_id: batch_users,
                self.item_id: batch_items,
                self.labels: batch_ratings.reshape(-1, 1),
            },
        )
        return _loss

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
                    self.gmf_user_id: [user_idx],
                    self.mlp_user_id: np.ones(self.train_set.num_items) * user_idx,
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
                feed_dict={
                    self.gmf_user_id: [user_idx],
                    self.mlp_user_id: [user_idx],
                    self.item_id: [item_idx],
                },
            )
            return user_pred.ravel()
