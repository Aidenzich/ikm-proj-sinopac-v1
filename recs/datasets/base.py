

import warnings
import numpy as np
from collections import OrderedDict, defaultdict
from scipy.sparse import csr_matrix, dok_matrix

from ..utils import get_random_state
from ..utils import estimate_number_batches


class Dataset(object):
    def __init__(
        self,
        num_users,
        num_items,
        uid_map,
        iid_map,
        uir_tuple,
        seed=None,
    ):

        self.uid_map = uid_map
        self.iid_map = iid_map
        self.num_users = num_users
        self.num_items = num_items
        self.uir_tuple = uir_tuple
        self.total_items = num_items
        self.seed = seed
        self.random_state = get_random_state(seed)

        (_, _, r_values) = uir_tuple

        self.num_ratings = len(r_values)
        self.max_rating = np.max(r_values)
        self.min_rating = np.min(r_values)
        self.global_mean = np.mean(r_values)

        self.__user_data = None
        self.__csr_matrix = None
        self.__dok_matrix = None

    @property
    def user_ids(self):
        return self.uid_map.keys()

    @property
    def item_ids(self):
        return self.iid_map.keys()

    @property
    def user_indices(self):
        return self.uid_map.values()

    @property
    def item_indices(self):
        return self.iid_map.values()

    @property
    def user_data(self):
        if self.__user_data is None:
            self.__user_data = defaultdict()
            for u, i, r in zip(*self.uir_tuple):
                u_data = self.__user_data.setdefault(u, ([], []))
                u_data[0].append(i)
                u_data[1].append(r)
        return self.__user_data

    @property
    def matrix(self):
        return self.csr_matrix

    @property
    def csr_matrix(self):
        if self.__csr_matrix is None:
            (u_indices, i_indices, r_values) = self.uir_tuple
            self.__csr_matrix = csr_matrix(
                (r_values, (u_indices, i_indices)),
                shape=(self.num_users, self.num_items),
            )
        return self.__csr_matrix

    @property
    def dok_matrix(self):
        if self.__dok_matrix is None:
            self.__dok_matrix = dok_matrix(
                (self.num_users, self.num_items), dtype=np.float32
            )
            for u, i, r in zip(*self.uir_tuple):
                self.__dok_matrix[u, i] = r
        return self.__dok_matrix

    @classmethod
    def build(
        cls,
        data,
        global_uid_map=None,
        global_iid_map=None,
        seed=None,
        exclude_unknowns=False,
    ):

        if global_uid_map is None:
            global_uid_map = OrderedDict()
        if global_iid_map is None:
            global_iid_map = OrderedDict()

        uid_map = OrderedDict()
        iid_map = OrderedDict()

        u_indices = []
        i_indices = []
        r_values = []
        valid_idx = []

        ui_set = set()  # avoid duplicate observations
        dup_count = 0

        for idx, (uid, iid, rating, *) in enumerate(data):
            if exclude_unknowns and (
                uid not in global_uid_map or iid not in global_iid_map
            ):
                continue

            if (uid, iid) in ui_set:
                dup_count += 1
                continue
            ui_set.add((uid, iid))

            uid_map[uid] = global_uid_map.setdefault(uid, len(global_uid_map))
            iid_map[iid] = global_iid_map.setdefault(iid, len(global_iid_map))

            u_indices.append(uid_map[uid])
            i_indices.append(iid_map[iid])
            r_values.append(float(rating))
            valid_idx.append(idx)

        if dup_count > 0:
            warnings.warn(
                "%d duplicated observations are removed!" % dup_count)

        if len(ui_set) == 0:
            raise ValueError("data is empty after being filtered!")

        uir_tuple = (
            np.asarray(u_indices, dtype=np.int),
            np.asarray(i_indices, dtype=np.int),
            np.asarray(r_values, dtype=np.float),
        )

        return cls(
            num_users=len(global_uid_map),
            num_items=len(global_iid_map),
            uid_map=uid_map,
            iid_map=iid_map,
            uir_tuple=uir_tuple,
            seed=seed,
        )

    @classmethod
    def from_uir(cls, data, seed=None):
        return cls.build(data, seed=seed)

    def reset(self):
        self.random_state = get_random_state(self.seed)
        return self

    def idx_iter(self, idx_range, batch_size=1, shuffle=False):
        indices = np.arange(idx_range)
        if shuffle:
            self.random_state.shuffle(indices)

        n_batches = estimate_number_batches(len(indices), batch_size)
        for b in range(n_batches):
            start_offset = batch_size * b
            end_offset = batch_size * b + batch_size
            end_offset = min(end_offset, len(indices))
            batch_ids = indices[start_offset:end_offset]
            yield batch_ids

    def uir_iter(self, batch_size=1, shuffle=False, binary=False, num_zeros=0):
        for batch_ids in self.idx_iter(len(self.uir_tuple[0]),
                                       batch_size, shuffle):
            batch_users = self.uir_tuple[0][batch_ids]
            batch_items = self.uir_tuple[1][batch_ids]
            if binary:
                batch_ratings = np.ones_like(batch_items)
            else:
                batch_ratings = self.uir_tuple[2][batch_ids]

            if num_zeros > 0:
                repeated_users = batch_users.repeat(num_zeros)
                neg_items = np.empty_like(repeated_users)
                for i, u in enumerate(repeated_users):
                    j = self.random_state.randint(0, self.num_items)
                    while self.dok_matrix[u, j] > 0:
                        j = self.random_state.randint(0, self.num_items)
                    neg_items[i] = j
                batch_users = np.concatenate((batch_users, repeated_users))
                batch_items = np.concatenate((batch_items, neg_items))
                batch_ratings = np.concatenate(
                    (batch_ratings, np.zeros_like(neg_items))
                )

            yield batch_users, batch_items, batch_ratings

    def user_iter(self, batch_size=1, shuffle=False):
        user_indices = np.fromiter(self.user_indices, dtype=np.int)
        for batch_ids in self.idx_iter(len(user_indices), batch_size, shuffle):
            yield user_indices[batch_ids]

    def item_iter(self, batch_size=1, shuffle=False):
        item_indices = np.fromiter(self.item_indices, np.int)
        for batch_ids in self.idx_iter(len(item_indices), batch_size, shuffle):
            yield item_indices[batch_ids]

    def is_unknown_user(self, user_idx):
        return user_idx >= self.num_users

    def is_unknown_item(self, item_idx):
        return item_idx >= self.num_items
