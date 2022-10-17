import numpy as np
import scipy.optimize as so
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd


#Currently not working
def curve_fit(f,x,y):
    return so.curve_fit(f,x,y)[0]

def min_squares(x,y):
    b_1 = np.sum((x-np.mean(x))*(y-np.mean(y)))/(np.sum(np.square((x-np.mean(x)))))

    b_0 = np.mean(y) -b_1*np.mean(x)

    return b_0,b_1

def std_err_est(x,y:np.ndarray,known_values = False,func = None,**kwargs):
    x = np.array(x)
    y = np.array(y)

    if not known_values:
        b_0,b_1 = min_squares(x,y)
    else:
        b_0 = kwargs.get("b0")
        b_1 = kwargs.get("b1")

    def line_eq(x_i,a,b):
        return x_i * float(a) + b

    if func == None:
        y_i = line_eq(x,b_1,b_0)
    else:
        y_i = func(x,b_1,b_0)

    return np.sqrt(np.sum((np.square(y-y_i)) / (y.size - 2)))

def lin_multivar_std_err_est(x:pd.DataFrame,y,params:list):
    #Note: params must be alligned to their corresponding predictors,
    #and params[0] MUST be the Constant term of the linear approximation.

    y = np.array(y)

    if type(x) != pd.DataFrame:
        raise TypeError("X must be a dataframe object.")

    if len(x.columns) != len(params-1):
        raise Exception("Number of Predictors and Coefficients Not Equal.")

    y_i = (params[0] + pd.DataFrame.sum(x*params[1:],axis="columns")).to_numpy()

    if y_i.size != y.size:
        raise Exception("Length of predicted Y values and Real Y do not match. Something went wrong.")

    return np.sqrt(np.sum((np.square(y-y_i)) / (y.size - len(params))))

def corr_pearson(x,y):
    return st.pearsonr(x,y)

def corr_spearman(x,y):
    return st.spearmanr(x,y)

def get_linreg_stats(data,x_columns,y_name):
    x = data[x_columns]

    x = sm.add_constant(x)

    y = data[y_name]

    results = sm.OLS(y,x).fit()
    print(results.summary())
    return results

def one_way_anova_fstat(*dists):
    return st.f_oneway(*dists)

def anova_smthng_idk(*dists):
    #Pass Dist object not Dist.dist.

    dist_list = [dst.dist for dst in dists]
    dist_list   
    y = np.concatenate(*dist_list)

    X = np.empty_like(y.size)


    i = 0
    j = 0
    for dst in dists:
        X[i:][1] = j

        i += dst.count-1
        j += 1

    df = pd.DataFrame((y,X), columns = ["y","X"])
    
    pseudo_code = "y ~ C(X)"

    model = ols(pseudo_code, data=df).fit()
    print(model.summary())
    aov_table = sm.stats.anova_lm(model, typ=1)

    print(aov_table)