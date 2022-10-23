from collections import defaultdict
import numpy as np
import statsmodels.stats.power as smp
import statsmodels.stats.multitest as smm
from scipy import stats as st
from .Dist import dist
from math import ceil
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
import pingouin as pg
from itertools import combinations

def cp_zscore(a,val):
    return (a.cp_mean() - val)/a.cp_std()

def cp_tscore(a,val,mode="zscore"):
    return cp_zscore(a,val)*10 + 50 if mode == 'zscore' else (a.cp_mean() - val)/a.cp_std_err()

def cp_f_stat(val,n):
    #Where val is the correlation coefficient and n is number of pairs.
    return (val**2 * (n-2))/(1- val**2)

def z_alpha(val,type = "tt"):
    if type == "lt":
        x = val
    elif type == "tt":
        x = 1-val/2
    elif type == "rt":
        x = 1-val
    
    return st.norm.ppf(x)


def t_alpha(alpha,df,type = "tt"):
    if type == "lt":
        x = alpha
    elif type == "tt":
        x = 1-alpha/2
    elif type == "rt":
        x = 1-alpha
    
    return st.t.ppf(x, df = df)

def cp_pval(val, df, *, right_tail = False, two_sided = False):
    if two_sided:
        return 2*st.t(df).cdf(val)
    elif right_tail:
        return 1 - st.t(df).cdf(val)
    else:
        return st.t(df).cdf(val)


def cp_fval(val,num_predictors,num_pairs):
    return 1 - st.f.cdf(val,num_predictors-1,num_pairs-num_predictors)


def cp_df(known_vals = False,**kwargs):

    if known_vals:
        a_var = kwargs.get("a_var")
        b_var = kwargs.get("b_var")

        a_count = kwargs.get("a_count")
        b_count = kwargs.get("b_count")  
    else:
        a = kwargs.get("a")
        b = kwargs.get("b")

        a_var = a.cp_var()
        b_var = b.cp_var()

        a_count = a.count
        b_count = b.count

    return np.square(a_var/a_count + b_var/b_count)/(a_var**2/(a_count**2 * (a_count-1)) + b_var**2/(b_count**2 * (b_count-1)))

def cp_1smp_cohen(a:dist,val:float)->float:
    return (a.cp_mean()-val)/a.cp_std()

def cp_2smp_cohen(known_values = False, **kwargs):
    if known_values:
        a_mean = kwargs.get("a_mean")
        b_mean = kwargs.get("b_mean")

        a_count = kwargs.get("a_count")
        b_count = kwargs.get("b_count")

        a_var = kwargs.get("a_var")
        b_var = kwargs.get("b_var")
    else:
        a = kwargs.get("a")
        b = kwargs.get("b")

        a_mean = a.cp_mean()
        b_mean = b.cp_mean()

        a_count = a.count
        b_count = b.count

        a_var = a.cp_var()
        b_var = b.cp_var()

    return (a_mean-b_mean)/np.sqrt((((a_count - 1) * a_var) + (b_count - 1) * b_var)/(a_count +  b_count - 2))

def multi_2smp_cohen(*dists):
    return [cp_2smp_cohen(a=a,b=b) for a,b in combinations(dists,2)]

def cp_comm_lang_eff(a,b):
    c = []
    for i in a:
        for j in b:
            if i < j:
                c.append(1)
            elif i == j:
                c.append(0.5)
            else:
                c.append(0)
    
    return (abs((sum(c))/(len(a) * len(b)) - 0.5) + 0.5) * 100
    

def cp_welchs_ci(x,alpha,known_vals = False,return_format = 'list',**kwargs):
    if known_vals:
        a_var = kwargs.get("a_var")
        b_var = kwargs.get("b_var")

        a_count = kwargs.get("a_count")
        b_count = kwargs.get("b_count")

    else:
        a = kwargs.get("a")
        b = kwargs.get("b")

        a_var = a.cp_var()
        b_var = b.cp_var()

        a_count = a.count
        b_count = b.count

    df = cp_df(True,a_var = a_var,b_var = b_var, a_count = a_count, b_count = b_count)
    t_score = t_alpha(alpha,df,"tt")

    tscore_times_stuff = t_score*np.sqrt(a_var/a_count + b_var/b_count)
    
    if return_format != 'list':
        plus_minus = u"\u00B1"
        return f"{x:.3f} {plus_minus} {tscore_times_stuff:.3f}"
    else:
        return [x - tscore_times_stuff, x + tscore_times_stuff]


def cp_stderr_mean_diff(a,b,known_var = False,**kwargs):
    if known_var:
        a_pop_var = kwargs.get("a_var")
        b_pop_var = kwargs.get("b_var")
        return np.sqrt(a_pop_var/a.count + b_pop_var/b.count)
    else:
        return np.sqrt(a.cp_var()/a.count + b.cp_var()/b.count)


def cp_tscore_diff_means(a,b,x:float = 0,known_var = False,return_z = False,**kwargs):
    #x is the value to be tested, i.e mean1 - mean2 = x

    if known_var:
        a_pop_var = kwargs.get("a_var")
        b_pop_var = kwargs.get("b_var")

        z = (a.cp_mean() - b.cp_mean() - x)/np.sqrt(a_pop_var/a.count + b_pop_var/b.count)
        
        if return_z:
            return z
        else:
            return z*10 + 50
    else:
        return (a.cp_mean()-b.cp_mean() - x)/np.sqrt(a.cp_var()/a.count + b.cp_var()/b.count)

def cp_pval_diff_means(a,b):
    return cp_pval(val = cp_tscore_diff_means(a=a,b=b), df= cp_df(a=a,b=b),two_sided=True)

def t_test_ind(a,b, *, eq_var = True, type = "two-sided"):
    return st.ttest_ind(a,b,equal_var=eq_var,alternative=type)

def t_test_1_samp(a,val, *, type = "two-sided", ret_values = True, **kwargs):
    #Type indicates the alternative hypothesis
    #options for type are 'two-sided', 'less'(than the indicated value), or 'greater'(than the indicated value.)

    t_stat,p_val = st.ttest_1samp(a,val,alternative=type)

    if ret_values:
        return [t_stat,p_val]
    else:
        #TODO:Finish #2 this to look more like R stats thing
        alpha = kwargs.get("alpha")
        print()

def t_test_paired(a,b, *, type = "two-sided", ret_values = True, **kwargs):
    t_stat,p_val = st.ttest_rel(a,b,alternative=type)
    #TODO: #3 Add stats printout if not ret_values(t-test-paired)

    if ret_values:
        return [t_stat,p_val]

#TODO: #4 This whole power function needs to be revaluated for numerical appx.
def t_test_pwr(alpha:float = 0.05, power:float = 0.8, *, type:str = "two-sided", brute_force = False, effect_size = None,cp_power = False, **kwargs):
    #Type indicates the alternative hypothesis
    #options for type are 'two-sided', 'smaller'(than the indicated value), or 'larger'(than the indicated value.)
    
    if effect_size != None or brute_force == True:
        effect_size = effect_size
    else:
        a = kwargs.get("a")
        val = kwargs.get("val")
        effect_size = cp_1smp_cohen(a,val)

    if cp_power:
        nobs = kwargs.get("nobs")
        return smp.ttest_power(effect_size=effect_size,nobs = nobs,alpha = alpha, alternative=type,df=nobs-1)

    if brute_force: #Doesn't Work

        min_nobs = 2
        max_nobs = 5000

        i_max = 10000

        a_mean = kwargs.get("a_mean")
        a_var = kwargs.get("a_var")
        a_std = np.sqrt(a_var)

        b_mean = kwargs.get("b_mean")
        b_var = kwargs.get("b_var")
        b_std = np.sqrt(b_var)

        Power = np.zeros(max_nobs-min_nobs)

        count = 0
        for n in range(min_nobs,max_nobs):
            p_values = np.zeros(i_max)

            

            for j in range(i_max):

                group_1 = dist(type='gaussian',mu=a_mean,sigma=a_std).dist.rvs(n)
                group_2 = dist(type='gaussian',mu=b_mean,sigma=b_std).dist.rvs(n)

                p_values[j] = t_test_ind(group_1,group_2)[1]

            
            arr_filter = p_values < alpha 

            Power[count] = np.sum(p_values[arr_filter])/i_max

            count += 1
        
        arr_filter = Power > power

        tmp_arr = Power[arr_filter]

        nobs = np.where(Power == tmp_arr[0]) + min_nobs

        return nobs

    obj = smp.TTestPower()
    return ceil(obj.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative=type))

def cp_tscore_paired(a:dist, val:float)->float:
    return (a.cp_mean() - val)/a.cp_std_err()

def cp_shapiro_wilk(dist):
    stat, p_val = st.shapiro(dist)
    return stat, p_val

def cp_multi_shapiro(*dists,alpha = 0.05):
    return all([False if p <= alpha else True for p in [cp_shapiro_wilk(dst)[1] for dst in dists]])

def cp_mann_whitney(a,b,*,alternative="two-sided",method="auto"):
    return st.mannwhitneyu(a,b,alternative=alternative,method=method)

def cp_wilcoxon(a:dist,b:dist=None,*,type = 'wilcox', alternative="two-sided"):
    return st.wilcoxon(a,b,zero_method= type, alternative=alternative)

def p_adjust(p_values:list,alpha:float = 0.05,*,method:str = "bonferroni",**kwargs):
    return smm.multipletests(pvals=p_values,alpha=alpha,method=method,**kwargs)

def cp_barlett(*dists,alpha = 0.05):
    return st.bartlett(*dists)[1] > alpha

def cp_bivar_cdf(dist,axis:str,val: float|list,ret_survival = False) -> float:
    #For this one, pass the actual Dist class instance as parameter, NOT Dist.dist
    if axis in ["x","a"]:
        a_mean = dist.a_mean
        a_std = np.sqrt(dist.a_var)

        return st.norm(a_mean,a_std).cdf(val) if not ret_survival else 1 - st.norm(a_mean,a_std).cdf(val)
    elif axis in ["y","b"]:
        b_mean = dist.b_mean
        b_std = np.sqrt(dist.b_var)

        return st.norm(b_mean,b_std).cdf(val) if not ret_survival else 1 - st.norm(b_mean,b_std).cdf(val)
    elif axis == 'both' and type(val) in [list,np.ndarray]:
        return dist.dist.cdf(val) if not ret_survival else 1 - dist.dist.cdf(val)

def cp_bivar_pdf(dist,axis:str,val: float|list) -> float:
    #For this one, pass the actual Dist class as parameter, NOT Dist.dist
    if axis in ["x","a"]:
        a_mean = dist.a_mean
        a_std = np.sqrt(dist.a_var)

        return st.norm(a_mean,a_std).pdf(val)
    elif axis in ["y","b"]:
        b_mean = dist.b_mean
        b_std = np.sqrt(dist.b_var)

        return st.norm(b_mean,b_std).pdf(val)
    elif axis == 'both' and type(val) in [list,np.ndarray]:
        return dist.dist.pdf(val)

def cp_bivar_cond_cdf(mean_a,mean_b,var_a,var_b,corr,x,y,flip_x = False):
    #Testing probability of X <= x, under constrain Y = Y.

    if not flip_x:
        mean = mean_a + np.sqrt(var_a/var_b)*corr*(y-mean_b)
        variance = (1-corr**2)*var_a

        return dist(type="gaussian",mu = mean, sigma = np.sqrt(variance)).dist.cdf(x)
    else:
        mean = mean_b + np.sqrt(var_b/var_a)*corr*(x-mean_a)
        variance = (1-corr**2)*var_b

        return dist(type="gaussian",mu = mean, sigma = np.sqrt(variance)).dist.cdf(y)

def one_way_anova_fstat(*dists):
    return st.f_oneway(*dists)

def omega_squared(aov):
    try:
        mse = aov['sum_sq'][-1]/aov['df'][-1]
        aov['omega_sq'] = 'NaN'
        aov['omega_sq'] = (aov[:-1]['sum_sq']-(aov[:-1]['df']*mse))/(sum(aov['sum_sq'])+mse)
    except KeyError:
        #TODO: #6 This works, but output is horrid.
        mse = aov.tail(1)["SS"]/aov.tail(1)["DF"]
        aov['omega_sq'] = 'NaN'
        a_sum = sum(aov['SS'])+mse
        for i in range(len(aov["SS"]-1)):
            aov['omega_sq'][i] = (aov["SS"][i] - aov["DF"][i]*mse)/a_sum
    
    return aov

def anova_smthng_idk(*dists):
    #Pass Dist object not Dist.dist.

    dist_list = [dst.dist for dst in dists]
     
    y = np.concatenate([*dist_list])

    X = np.empty_like(y)

    

    #Pretty Sure this will work?
    i = 0
    j = 0
    for dst in dists:
        for k in range(dst.count):
            X[i+k] = j

        i += dst.count
        j += 1

    df = pd.DataFrame([y,X])
    
    df = df.transpose()

    df.columns = ["y","X"]

    pseudo_code = "y ~ C(X)"

    model = ols(pseudo_code, data=df).fit()
    print(model.summary())
    aov_table = sm.stats.anova_lm(model, typ=2)

    aov_table = omega_squared(aov_table)

    print(aov_table)

#TODO: #5 Figure out how to implement this
def dists_to_df(dists):
    tmp_dict = defaultdict()
    for dst in dists:
        tmp_dict[dst.name] = []
        for item in dst.dist:
            tmp_dict[dst.name].append(item)