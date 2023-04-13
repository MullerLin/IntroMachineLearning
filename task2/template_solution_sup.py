# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score


def oneHot_encoding(data:pd.DataFrame) -> pd.DataFrame:
    N = data.shape[0]
    season_encoding_ndarry = np.zeros((N, 4))
    seasons = ['spring', 'summer', 'autumn', 'winter']

    for i in range(N):
        season = [j for j in range(4) if seasons[j] == data['season'][i]]
        assert(len(season) == 1)
        season_encoding_ndarry[i][season[0]] = 1

    season_encoding_df = pd.DataFrame(data=season_encoding_ndarry, columns=seasons)
    price_df = data.drop(['season'],axis=1)
    encoded_data_df = pd.concat([season_encoding_df, price_df], axis=1)
    return encoded_data_df


def Matern_para_optim(X, y, n_folds):
    print("____________Matern___________")
    params = [  (0.05, 0.5), (0.05, 1.5), (0.05, 2.5),
                (0.1, 0.5), (0.1, 1.5), (0.1, 2.5),
                (1, 0.5), (1, 1.5), (1, 2.5)]
    R2score_mat = np.zeros((n_folds, len(params)))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=14)
    for fold_idx, (train, test) in enumerate(kf.split(X)):
        X_train, X_val, y_train, y_val = X[train], X[test], y[train], y[test]
        for idx, (ls, nu) in enumerate(params):
            gpr = GaussianProcessRegressor(kernel = Matern(length_scale=ls, nu=nu))
            gpr.fit(X_train, y_train)
            y_val_pred = gpr.predict(X_val)
            R2score_mat[fold_idx][idx] = r2_score(y_val, y_val_pred)

    avg_R2score = np.mean(R2score_mat, axis=0)
    print(avg_R2score)
    print("______________end_____________")




def RBF_para_optim(X, y, n_folds):
    print("____________RBF___________")
    lambdas = [0.05, 0.1, 1]
    R2score_mat = np.zeros((n_folds, len(lambdas)))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=41)
    for fold_idx, (train, test) in enumerate(kf.split(X)):
        X_train, X_val, y_train, y_val = X[train], X[test], y[train], y[test]
        for idx, (ls) in enumerate(lambdas):
            gpr = GaussianProcessRegressor(kernel = RBF(length_scale=ls))
            gpr.fit(X_train, y_train)
            y_val_pred = gpr.predict(X_val)
            R2score_mat[fold_idx][idx] = r2_score(y_val, y_val_pred)

    avg_R2score = np.mean(R2score_mat, axis=0)
    print(avg_R2score)
    print("______________end_____________")



def RationalQuadratic_para_optim(X, y, n_folds):
    print("____________RationalQuadratic___________")
    params = [  (0.1, 0.1), (0.1, 1),
                (1, 0.1), (1, 1)]
    R2score_mat = np.zeros((n_folds, len(params)))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=14)
    for fold_idx, (train, test) in enumerate(kf.split(X)):
        X_train, X_val, y_train, y_val = X[train], X[test], y[train], y[test]
        for idx, (ls, al) in enumerate(params):
            gpr = GaussianProcessRegressor(kernel = RationalQuadratic(length_scale=ls, alpha=al))
            gpr.fit(X_train, y_train)
            y_val_pred = gpr.predict(X_val)
            R2score_mat[fold_idx][idx] = r2_score(y_val, y_val_pred)

    avg_R2score = np.mean(R2score_mat, axis=0)
    print(avg_R2score)

    print("______________end_____________")

        
        

def RQM_sup_para_select(X, y, n_folds):
    print("————Rational Quadratic + Matern————")
    RQ_params = [  (0.1, 0.1), (0.1, 1),
                (1, 0.1), (1, 1)]
    M_params = [  (0.05, 0.5), (0.05, 1.5), (0.05, 2.5),
                (0.1, 0.5), (0.1, 1.5), (0.1, 2.5),
                (1, 0.5), (1, 1.5), (1, 2.5)]
    params = []
    for i in range(len(RQ_params)):
        for j in range(len(M_params)):
            params.append((RQ_params[i], M_params[j]))
    R2score_mat = np.zeros((n_folds, len(params)))    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state= 14)
    for fold_idx, (train, test) in enumerate(kf.split(X)):
        X_train, X_val, y_train, y_val = X[train], X[test], y[train], y[test]
        for idx, (RQ, M) in enumerate(params):
            kernel = RationalQuadratic(length_scale=RQ[0], alpha=RQ[1]) + Matern(length_scale=M[0], nu=M[1])
            gpr = GaussianProcessRegressor(kernel = kernel)
            gpr.fit(X_train, y_train)
            y_val_pred = gpr.predict(X_val)
            R2score_mat[fold_idx][idx] = r2_score(y_val, y_val_pred)
            
    avg_R2score = np.mean(R2score_mat, axis=0)
    print("Best Param: ", params[np.argmax([avg_R2score])], " Score: ", avg_R2score[np.argmax([avg_R2score])])
    print(avg_R2score)
    print("______________end_____________")
    return avg_R2score, params[np.argmax([avg_R2score])]     




def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))


    # One-hot encoding
    encoded_train_df = oneHot_encoding(train_df)
    encoded_test_df = oneHot_encoding(test_df)

    # encoded_train_df.to_csv('encoded.csv', index=False)

    imputer = KNNImputer(n_neighbors=10, weights="uniform")

    imputed_train = imputer.fit_transform(encoded_train_df)
    imputed_test = imputer.fit_transform(encoded_test_df)

    train_idx = [i for i in range(encoded_train_df.shape[0]) if np.isnan(encoded_train_df['price_CHF'][i]) == False]
    # print('length: ', len(train_idx))

    X_train_raw = np.delete(imputed_train, 5, 1)
    y_train_raw = imputed_train[:, 5]

    # dt = pd.DataFrame(y_train_raw, columns=['label'])
    # dt.to_csv('y_train_raw.csv', index=False)

    # Dummy initialization of the X_train, X_test and y_train   
    X_train = X_train_raw.take(train_idx, 0)
    y_train = y_train_raw.take(train_idx, 0)
    X_test = imputed_test
    
    # X_train = np.zeros_like(encoded_train_df.drop(['price_CHF'],axis=1))
    # y_train = np.zeros_like(encoded_train_df['price_CHF'])
    # X_test = np.zeros_like(encoded_test_df)
    

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    y_pred=np.zeros(X_test.shape[0])
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions


    # Model selection
    kernels = [ RBF, Matern, RationalQuadratic]
    N = len(kernels)
    n_folds = 10
    # R2score_mat = np.zeros((n_folds, N))

    # kf = KFold(n_splits=n_folds, shuffle=True, random_state=41)
    # for fold_idx, (train, test) in enumerate(kf.split(X_train)):
    #     X_train_val, X_val, y_train_val, y_val = X_train[train], X_train[test], y_train[train], y_train[test]
    #     for func in range(N):
    #         gpr = GaussianProcessRegressor(kernel = kernels[func]())
    #         gpr.fit(X_train_val, y_train_val)
    #         y_val_pred = gpr.predict(X_val)
    #         R2score_mat[fold_idx][func] = r2_score(y_val, y_val_pred)

    # avg_R2score = np.mean(R2score_mat, axis=0)
    # print(avg_R2score)

    # Matern_para_optim(X_train, y_train, n_folds)
    # RBF_para_optim(X_train, y_train, n_folds)
    # RationalQuadratic_para_optim(X_train, y_train, n_folds)


    kernel= RationalQuadratic(length_scale=1, alpha=0.1) * Matern(length_scale=0.05, nu=1.5)
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)
    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    # print("\nResults file successfully generated!")

