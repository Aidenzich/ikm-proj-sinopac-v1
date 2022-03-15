import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

class MatrixFactorization:
    """Matrix Factorization
        from: Matrix Factorization Techniques for Recommender Systems

    Parameters
    ----------
    train_df: pd.DataFrame, Training DataFrame, with columns = [pivot_index_name, pivot_columns_name, pivot_values_name]
    svds_k: int, Hyperparameter K for svds
    pivot_index_name: str, user columns name
    pivot_columns_name: str, item columns name
    pivot_values_name: str, rank columns name
    """
    def __init__(self, train_df, svds_k,
                 pivot_index_name = "user",
                 pivot_columns_name = "item",
                 pivot_values_name = "rank"                                  
        ):
        
        self.name = "MatrixFactorization"
        self.result_df = pd.DataFrame()
        self.svds_k = svds_k
        self.pivot_index = pivot_index_name
        self.pivot_columns = pivot_columns_name
        self.pivot_values = pivot_values_name
        self.fit(train_df)
    
    def fit(self, train_df):
        u_i_p_matrix_df = train_df.pivot(
            index=self.pivot_index, 
            columns=self.pivot_columns, 
            values=self.pivot_values
        ).fillna(0)
        
        u_i_p_matrix = u_i_p_matrix_df.values
        u_i_p_sparse_matrix = csr_matrix(u_i_p_matrix)
        
        # do svd
        U, sigma, Vt = svds(u_i_p_sparse_matrix, k=self.svds_k)
        sigma = np.diag(sigma)
        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        
        # normalize
        all_user_predicted_ratings_norm = (all_user_predicted_ratings - all_user_predicted_ratings.min()
                                        ) / (all_user_predicted_ratings.max() - all_user_predicted_ratings.min())
        
        # result
        self.result_df = pd.DataFrame(all_user_predicted_ratings_norm, columns=u_i_p_matrix_df.columns, 
                                index=list(u_i_p_matrix_df.index)).transpose()
    
    
    def rec_items(self, user, items_to_ignore=[], topn=10):
        """recommender items by user's id
        Parameters
        ----------
        user: str, user's id.
        items_to_ignore: list, list of items which you want to ignore.
        topn: int, The number of items you want to recommend.
        """
        sorted_user_preds = self.result_df[user].sort_values(ascending=False).reset_index() \
            .rename(columns={user:self.pivot_values})
        
        rec_df = sorted_user_preds[~sorted_user_preds[self.pivot_columns].isin(items_to_ignore)
            ].sort_values(self.pivot_values, ascending=False).head(topn)
        
        return rec_df


