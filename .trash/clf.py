from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier,
RandomForestClassifier, AdaBoostClassifier, BaggingRegressor, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class Classifier:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = None
        self.model_name = "No training model"
        
    def do_bagginregression(self, n_estimators=100, random_state=0)->None:
        self.model = BaggingRegressor(base_esimator=SVR(),
                                     n_esimators = n_estimators,
                                     random_state=0)
        self.model.fit(self.x_train, self.y_train)
        self.model_name = "BaggingRegressor"
    
    def do_lgbm(self, 
                n_estimators=100, random_state=42, 
                bagging_fraction=0.67, feature_fraction=0.06,
                bagging_freq=1,
                verbose=1, n_jobs=6,
                max_depth=3,
                num_leaves = 31,
                boosting_type = "gbdt"
               ):
        self.model = LGBMClassifier(
            boosting_type = boosting_type,
            n_estimators=n_estimators,
            random_state = random_state,
            bagging_fraction = bagging_fraction,
            feature_fraction = feature_fraction,
            bagging_freq = bagging_freq,
            verbose = verbose,
            n_jobs= n_jobs,
            max_depth=max_depth,
            num_leaves = num_leaves,
        )
        
        self.model.fit(self.x_train, self.y_train)
        self.model_name = "LGBMClassifier"
    
    def do_adaboost(self, 
                    n_estimators=100, 
                    random_state=0
                ) -> None:
        self.model = AdaBoostClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )
        self.model.fit(self.x_train, self.y_train)
        self.model_name = "AdaBoostClassifier"
        
    def do_logistic(self) -> None:
        print('Start LogisticRegression Training...')
        self.model = LogisticRegression(
            random_state=0,
            verbose=1,
            max_iter=1000            
        )
        self.model.fit(self.x_train, self.y_train)
        self.model_name = "LogisticRegression"
        
    def do_gradient_boosting(self, n_estimators=100, 
                             lr=0.1, max_depth=2, random_state=42) -> None:
        
        print('Start Gradient Boosting Training...')
        
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=lr, 
            max_depth=max_depth, 
            random_state=random_state,
            verbose=1
        )
        self.model.fit(self.x_train, self.y_train)
        self.model_name = "GradientBoostingClassifier"
        
    def do_xgboost(self, 
                   n_estimators=100, lr=0.1, max_depth=2, 
                   random_state=42):
        
        print('Training XGBClassifier')
        
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=lr, 
            max_depth=max_depth, 
            random_state=random_state,
            verbosity=1
        )
        self.model.fit(self.x_train, self.y_train)
        self.model_name = "XGBClassifier"
        
    def do_random_forest(self, n_estimators=100, max_depth=2, random_state=0):
        print('Training RandomForestClassifier')
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth, 
            random_state=random_state,
            verbose=1
        )
        self.model.fit(self.x_train, self.y_train)
        self.model_name = "RandomForestClassifier"
        
    def do_stacking(self, n_estimators=100, max_depth=2, random_state=0):
        estimators = [
            ('rf', RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth, 
                random_state=random_state,         
            )),
            ('ab', AdaBoostClassifier(
                n_estimators=n_estimators,
                random_state=random_state
            )),
            ('gb',  GradientBoostingClassifier(
                n_estimators=n_estimators,                
                max_depth=max_depth, 
                random_state=random_state,            
            )),
            ('svr', make_pipeline(
                StandardScaler(),
                LinearSVC(random_state=42)
            )),                        
        ]
        
        self.model = StackingClassifier(
            estimators=estimators, final_estimator= LogisticRegression(
                random_state=0,
                verbose=1,
                max_iter=1000   
            ),
            n_jobs = 8, passthrough=True
        )
        self.model.fit(self.x_train, self.y_train)
        self.model_name = "StackingClassifier"
        
        
    
    def check_score(self) -> bool:
        if self.model == None:
            return False
        y_predict = self.model.predict(self.x_test)
        self.y_predict = y_predict
        _model_score = self.model.score(self.x_train, self.y_train)
        _accuracy_score = accuracy_score(self.y_test, y_predict)
        _f1_score = f1_score(self.y_test, y_predict, average='binary')        
        _precision_score = precision_score(self.y_test, y_predict)
        _recall_score = recall_score(self.y_test, y_predict)
        self.score_result = {
            "model_score": _model_score,
            "accuracy_score": _accuracy_score,
            "f1_score": _f1_score,
            "precision_score": _precision_score,
            "recall_score": _recall_score            
        }
        
        return True

