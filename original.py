import pandas as pd
from scipy.optimize import fsolve
from sklearn.linear_model import LogisticRegression
import scipy as sp
import numpy as np
import random
import logging
import argparse
import os
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser(description='Fairness Boosting Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--rs', '--random-state', default=None,
                    metavar='RS', help='random state for train-test split')



def newtown(func, x0, x1, res, pro, ari, mode="right"):
    '''
    The fsolve function in sp.optimize always return some strange solutions. therefore
    I write this newtown method to calculate the root of the function
    :param func: score function
    :param x0: initial guess of the root
    :param x1: another initial guess of the root
    :param res: true result of each records
    :param pro: protected attribute
    :param ari: arity of the protected attribute
    :param mode: right=underestimate, left=overestimate
    :return: The root of func
    '''
    num_iter = 0
    if (mode == "right"):
        while (abs(x1 - x0) > 0.00000001):
            x_tmp = x1 - func(x1, res, pro, ari) * (x1 - x0) / (func(x1, res, pro, ari) - func(x0, res, pro, ari))
            x0 = x1
            x1 = x_tmp
            num_iter += 1
            if (num_iter > MAX_ITER):
                break
    else:
        while (abs(x1 - x0) > 0.00000001):
            x_tmp = x1 - func(x1, res, pro, ari) * (x1 - x0) / (func(x1, res, pro, ari) - func(x0, res, pro, ari))
            x0 = x1
            x1 = x_tmp
            if (x_tmp < 0.00000001):
                x_tmp = 0.00000001
            num_iter += 1
            if (num_iter > MAX_ITER):
                break
    return x1



def solve_q_max(res, pro, ari, mode="right", root=True):
    '''
    For the given subset, estimate the best q
    :param res: true result of each records
    :param pro: predicted probability of each records
    :param ari: arity of the protected attribute
    :param mode: right=underestimate, left=overestimate
    :param root: if return the only q_max or both q_max and q_0
    :return: q_max or both q_max and q_0
    '''
    if (mode == "left"):
        q_max = 0.6
        for ini in [0.1, 0.5, 0.9]:
            #multiply initial values to avoid local optimal
            q_max_tmp = sp.optimize.minimize(func, ini, args=(res, pro, ari), bounds=[(0, 1)]).x[0]
            if (func(q_max_tmp, res, pro, ari) < func(q_max, res, pro, ari)):
                q_max = q_max_tmp
        if (q_max > 1):
            q_max = 1
        if (not root):
            return q_max
        else:
            if (q_max == 1):
                return q_max, q_max, q_max
            k = 1
            q_0 = newtown(func, 0.9 * q_max, 0.8 * q_max, res, pro, ari, mode)
            if (np.isnan(q_0)):
                q_0 = sp.optimize.fsolve(func, 0.9 * q_max, (res, pro, ari))[0]
            if (q_0 == 1):
                q_0 = q_max
            q_1 = newtown(func, 1, 2 * q_max - q_0, res, pro, ari, mode)
            if (np.isnan(q_1)):
                q_1 = sp.optimize.fsolve(func, 2 * q_max - q_0, (res, pro, ari))[0]
            return q_max, q_0, q_1
    elif (mode == "right"):
        q_max = 1
        for ini in [1, 1.5, 3, 5]:
            q_max_tmp = sp.optimize.minimize(func, ini, args=(res, pro, ari), bounds=[(1, None)]).x[0]
            if (func(q_max_tmp, res, pro, ari) < func(q_max, res, pro, ari)):
                q_max = q_max_tmp
        if (q_max < 1):
            q_max = 1
        if (not root):
            return q_max
        else:
            if (q_max == 1):
                return q_max, q_max, q_max
            # print(q_max)
            k = 1
            q_0 = newtown(func, q_max, 2 * q_max, res, pro, ari, mode)
            if (np.isnan(q_0)):
                q_0 = sp.optimize.fsolve(func, 2 * q_max, (res, pro, ari))[0]
            if (q_0 == 1):
                q_0 = q_max
            q_1 = newtown(func, 1, 2 * q_max - q_0, res, pro, ari, mode)

            if (np.isnan(q_1)):
                q_1 = sp.optimize.fsolve(func, 2 * q_max - q_0, (res, pro, ari))[0]

            return q_max, q_0, q_1
    else:
        q_max = 1
        for ini in [0.1, 0.5, 0.9, 1.5, 5]:
            q_max_tmp = sp.optimize.minimize(func, ini, args=(res, pro, ari), bounds=[(0, None)]).x[0]
            if (func(q_max_tmp, res, pro, ari) < func(q_max, res, pro, ari)):
                q_max = q_max_tmp
        return q_max


def func(x, res, pro, arity):
    '''
    Score function of a given subset with the estimated q
    :param x: estimated q
    :param res: true result of each records
    :param pro: predicted probability of each records
    :param arity: arity of the protected attribute
    :return: Score function of a given subset with the estimated q
    '''
    return -(np.log(x) * sum(res) - sum(np.log(1 - pro + x * pro)) - PEL_RATE * arity)



def ALTSS(results, proba, protect, full_arity, mode="right"):
    '''
    Use ALTSS algorithm to dected the subset corresponding to one dimension
    :param results: true result of each records
    :param proba: predicted probability of each records
    :param protect:  predicted probability of each records
    :param full_arity:  total arity of the protected attribute
    :param mode: right=underestimate, left=overestimate
    :return: cur_max_score: best score corresponding to the detected subset
              best_subset: detected subset
    '''
    q_0_s = []
    q_1_s = []
    protect_val = np.unique(protect)
    if (mode == "right"):
        for i in protect_val:
            res = results[protect == i]
            pro = proba[protect == i]
            if (len(res) == len(res[res == 1])):
                q_0 = np.inf
                q_1 = np.inf
            elif (len(res) == len(res[res == 0])):
                q_0 = 1
                q_1 = 1
            else:
                _, q_0, q_1 = solve_q_max(res, pro, 1 / full_arity, mode)
            q_0_s.append(max(q_0, q_1))
            q_1_s.append(min(q_0, q_1))
        q_0_s = np.array(q_0_s)
        q_1_s = np.array(q_1_s)
        q_s_sort = np.sort(np.unique(np.concatenate([q_0_s, q_1_s])))
        subset = protect_val
        best_subset = subset
        pro = np.array(proba)
        res = np.array(results)
        max_q = sp.optimize.minimize(func, 1, args=(res, pro, 0), bounds=[(1, None)]).x[0]
        cur_max_score = -func(max_q, res, pro, 0)
        for i in range(1, len(q_s_sort)):
            subset = protect_val[(q_0_s >= q_s_sort[i]) & (q_1_s <= q_s_sort[i - 1])]
            ari = len(protect_val[(q_0_s >= q_s_sort[i]) & (q_1_s <= q_s_sort[i - 1])])
            res = np.array([results[i] for i in range(len(results)) if protect[i] in subset])
            if (len(res[res == 1]) == len(res)):
                pro = np.array([proba[i] for i in range(len(results)) if protect[i] in subset])
                max_score = -sum(np.log(pro)) - PEL_RATE * ari / full_arity
            else:
                pro = np.array([proba[i] for i in range(len(results)) if protect[i] in subset])
                max_q = sp.optimize.minimize(func, (q_s_sort[i - 1] + q_s_sort[i]) / 2, args=(res, pro, ari),
                                             bounds=[(q_s_sort[i - 1], q_s_sort[i])]).x[0]
                max_score = -func(max_q, res, pro, ari / full_arity)
            if (max_score > cur_max_score):
                cur_max_score = max_score
                best_subset = subset
        return cur_max_score, best_subset
    else:
        for i in protect_val:
            res = results[protect == i]
            pro = proba[protect == i]
            if (len(res) == len(res[res == 1])):
                q_0 = 1
                q_1 = 1
            elif (len(res) == len(res[res == 0])):
                q_0 = 0
                q_1 = 0
            else:
                _, q_0, q_1 = solve_q_max(res, pro, 1 / full_arity, mode)
            q_0_s.append(min(q_0, q_1))
            q_1_s.append(max(q_0, q_1))
        q_0_s = np.array(q_0_s)
        q_1_s = np.array(q_1_s)
        q_s_sort = np.sort(np.unique(np.concatenate([q_0_s, q_1_s])))
        subset = protect_val
        best_subset = subset
        pro = np.array(proba)
        res = np.array(results)
        max_q = sp.optimize.minimize(func, 0.5, args=(res, pro, 0), bounds=[(0, 1)]).x[0]
        cur_max_score = -func(max_q, res, pro, 0)
        for i in range(1, len(q_s_sort)):
            subset = protect_val[(q_0_s <= q_s_sort[i - 1]) & (q_1_s >= q_s_sort[i])]
            ari = len(protect_val[(q_0_s <= q_s_sort[i - 1]) & (q_1_s >= q_s_sort[i])])
            res = np.array([results[i] for i in range(len(results)) if protect[i] in subset])
            if (len(res[res == 1]) == len(res)):
                pro = np.array([proba[i] for i in range(len(results)) if protect[i] in subset])
                max_score = -sum(np.log(pro)) - PEL_RATE * ari / full_arity
            else:
                pro = np.array([proba[i] for i in range(len(results)) if protect[i] in subset])
                max_q = sp.optimize.minimize(func, (q_s_sort[i - 1] + q_s_sort[i]) / 2, args=(res, pro, ari),
                                             bounds=[(q_s_sort[i - 1], q_s_sort[i])]).x[0]
                max_score = -func(max_q, res, pro, ari / full_arity)
            if (max_score > cur_max_score):
                cur_max_score = max_score
                best_subset = subset
        return cur_max_score, best_subset



def bias_subscan_sub(dat, mode="right"):
    '''
    :param multi dimensional dataset to detecte
    :param mode: right=underestimate, left=overestimate
    :return: detected subgroup and its score
    '''
    best_score = -1
    cur_best_score = -np.inf
    protect_feature = list(dat.columns[:-2])
    feature_filter = {}
    last_feature_filter = {}
    for i in protect_feature:
        feature_filter[i] = dat[i].unique()
        last_feature_filter[i] = dat[i].unique()
    num_iter = 0
    while (cur_best_score != best_score):
        num_iter += 1
        if (num_iter > 100):
            break
        np.random.shuffle(protect_feature)
        best_score = cur_best_score
        for i in protect_feature:
            full_arity = len(dat[i].unique())
            feature_filter[i] = dat[i].unique()
            cur_data_subset = dat.loc[:, :]
            for k in protect_feature:
                cur_data_subset = cur_data_subset.loc[cur_data_subset[k].isin(feature_filter[k]), :]
            result = cur_data_subset['SeriousDlqin2yrs'].values
            proba = cur_data_subset['proba'].values
            protect = cur_data_subset[i].values
            cur_best_score, curs_subgroup = ALTSS(result, proba, protect, full_arity, mode)
            feature_filter[i] = curs_subgroup
        inx = 0
        for i in protect_feature:
            if (set(feature_filter[i]) == set(last_feature_filter[i])):
                inx += 1
        if (inx == len(protect_feature)):
            break
        last_feature_filter = feature_filter.copy()
    cur_data_subset = dat.loc[:, :]
    ari = 0
    for k in protect_feature:
        cur_data_subset = cur_data_subset.loc[cur_data_subset[k].isin(feature_filter[k]), :]
        if (len(feature_filter[k]) != len(dat[k].unique())):
            ari += len(feature_filter[k]) / len(dat[k].unique())
    res = cur_data_subset[ 'SeriousDlqin2yrs'].values
    pro = cur_data_subset['proba'].values
    if (mode == "right"):
        if (len(res) == len(res[res == 1])):
            q_max = np.inf
        elif (len(res) == len(res[res == 0])):
            q_max = 1
        else:
            q_max, _, _ = solve_q_max(res, pro, ari, mode)
    if (mode == "left"):
        if (len(res) == len(res[res == 1])):
            q_max = 1
        elif (len(res) == len(res[res == 0])):
            q_max = 0
        else:
            q_max, _, _ = solve_q_max(res, pro, ari, mode)
    cur_best_score = -func(q_max, res, pro, ari)


    return q_max, cur_best_score, feature_filter

def unpel_score(dat, filt, mode='right'):
    '''
    calculate the unpanelized score of a dataset corresponding to a give subset
    :param dat: dataset to calculate the score
    :param filt: a dict store the definition of the subset
    :param mode: right=underestimate, left=overestimate
    :return: unpanelized score of a dataset corresponding to a give subset
    '''

    protect_feature = filt.keys()
    cur_data_subset = dat
    for k in protect_feature:
        cur_data_subset = cur_data_subset.loc[cur_data_subset[k].isin(filt[k]), :]
    res = cur_data_subset['SeriousDlqin2yrs'].values
    pro = cur_data_subset['proba'].values
    if (mode == 'right'):
        if (len(res) == len(res[res == 1])):
            q_max_right = np.inf
            score_unpel = -sum(np.log(pro))
        elif (len(res) == len(res[res == 0])):
            q_max_right = 1
            score_unpel = -func(q_max_right, res, pro, 0)
        else:
            q_max_right, _, _ = solve_q_max(res, pro, 0, "right")
            score_unpel = -func(q_max_right, res, pro, 0)
    else:
        if (len(res) == len(res[res == 1])):
            q_max_right = 1
        elif (len(res) == len(res[res == 0])):
            q_max_right = 0
        else:
            q_max_right, _, _ = solve_q_max(res, pro, 0, "left")

        score_unpel = -func(q_max_right, res, pro, 0)

    return q_max_right, score_unpel


def bias_subscan(dat, mode="right"):
    '''
    multiply start with different initial condition to avoid local optimal
    :param dat: dataset to calculate the score
    :param mode: right=underestimate, left=overestimate
    :return: detected subgroup and its score
    '''
    best_score = -np.inf
    for i in range(5):
        q_max, cur_best_score, feature_filter = bias_subscan_sub(dat, mode)
        if (cur_best_score > best_score):
            best_score = cur_best_score
            best_q = q_max
            best_filter = feature_filter

    return best_q, best_score, best_filter

def randomize_test(pro, score, mode):
    sim_round = 1000
    score_sim = np.zeros(sim_round)
    PEL_RATE = 0
    for i in range(sim_round):
        res_sim = np.random.binomial(1, p = pro)
        q_sim = solve_q_max(res_sim, pro, 0, mode=mode, root=False)
        score_sim[i] = -func(q_sim, res_sim, pro, 0)
    return len(score_sim[score_sim >= score]) / sim_round


def bias_detection(dat, dat_test):
    global PEL_RATE

    PEL_RATE = PEL_RATE_ORI * len(dat) / 7214

    _, _, feature_filter_right = bias_subscan(dat, "right")
    _, _, feature_filter_left = bias_subscan(dat, "left")

    _, train_score_under = unpel_score(dat, feature_filter_right, mode='right')
    filt_right = (dat['RevolvingUtilizationOfUnsecuredLines'].isin(
        feature_filter_right['RevolvingUtilizationOfUnsecuredLines'])) \
                 & (dat['age'].isin(feature_filter_right['age'])) \
                 & (dat['NumberOfTime30-59DaysPastDueNotWorse'].isin(
        feature_filter_right['NumberOfTime30-59DaysPastDueNotWorse'])) \
                 & (dat['DebtRatio'].isin(feature_filter_right['DebtRatio'])) \
                 & (dat['MonthlyIncome'].isin(feature_filter_right['MonthlyIncome'])) \
                 & (dat['NumberOfOpenCreditLinesAndLoans'].isin(
        feature_filter_right['NumberOfOpenCreditLinesAndLoans'])) \
                 & (dat['NumberOfTimes90DaysLate'].isin(feature_filter_right['NumberOfTimes90DaysLate'])) \
                 & (dat['NumberRealEstateLoansOrLines'].isin(feature_filter_right['NumberRealEstateLoansOrLines'])) \
                 & (dat['NumberOfTime60-89DaysPastDueNotWorse'].isin(
        feature_filter_right['NumberOfTime60-89DaysPastDueNotWorse'])) \
                 & (dat['NumberOfDependents'].isin(feature_filter_right['NumberOfDependents']))

    pro = dat.loc[filt_right, 'proba'].values
    p_right = randomize_test(pro, train_score_under, mode='right')

    _, train_score_over = unpel_score(dat, feature_filter_left, mode='left')
    filt_left = (dat['RevolvingUtilizationOfUnsecuredLines'].isin(
        feature_filter_left['RevolvingUtilizationOfUnsecuredLines'])) \
                & (dat['age'].isin(feature_filter_left['age'])) \
                & (dat['NumberOfTime30-59DaysPastDueNotWorse'].isin(
        feature_filter_left['NumberOfTime30-59DaysPastDueNotWorse'])) \
                & (dat['DebtRatio'].isin(feature_filter_left['DebtRatio'])) \
                & (dat['MonthlyIncome'].isin(feature_filter_left['MonthlyIncome'])) \
                & (dat['NumberOfOpenCreditLinesAndLoans'].isin(feature_filter_left['NumberOfOpenCreditLinesAndLoans'])) \
                & (dat['NumberOfTimes90DaysLate'].isin(feature_filter_left['NumberOfTimes90DaysLate'])) \
                & (dat['NumberRealEstateLoansOrLines'].isin(feature_filter_left['NumberRealEstateLoansOrLines'])) \
                & (dat['NumberOfTime60-89DaysPastDueNotWorse'].isin(
        feature_filter_left['NumberOfTime60-89DaysPastDueNotWorse'])) \
                & (dat['NumberOfDependents'].isin(feature_filter_left['NumberOfDependents']))
    pro = dat.loc[filt_left, 'proba'].values
    p_left = randomize_test(pro, train_score_over, mode='left')
    print("over train = {} with p_value {}, under train = {} with p value {}".format(train_score_over,
                                                                                 p_right,
                                                                                 train_score_under,
                                                                                 p_left))

    _, test_score_over = unpel_score(dat_test, feature_filter_left, mode='left')
    _, test_score_under = unpel_score(dat_test, feature_filter_right, mode='right')

    print(
        "estimated over test ={}, estimated under test ={}".format(test_score_over, test_score_under))

    PEL_RATE = 0
    _, _, test_filt_right = bias_subscan(dat_test, "right")
    _, test_score_under = unpel_score(dat_test, test_filt_right, mode='right')
    _, _, test_filt_left = bias_subscan(dat_test, "left")
    _, test_score_over = unpel_score(dat_test, test_filt_left, mode='left')

    print(
        "true over test ={}, true under test ={}".format(test_score_over, test_score_under))
    return None



if __name__ == "__main__":

    args = parser.parse_args()

    print('Args:', args)

    PEL_RATE_ORI = 5
    MAX_ITER = 100
    csv_file_vr = args.data
    dat = pd.read_csv(csv_file_vr, index_col=0)
    #data processing
    dat.loc[dat['NumberOfDependents'].isnull(), 'NumberOfDependents'] = 0
    dat.loc[dat['MonthlyIncome'].isnull(), 'MonthlyIncome'] = dat['MonthlyIncome'].mean()
    dat.loc[dat['NumberOfTime30-59DaysPastDueNotWorse'].isin([96, 98]), 'NumberOfTime30-59DaysPastDueNotWorse'] = 0
    dat.loc[dat['NumberOfTimes90DaysLate'].isin([96, 98]), 'NumberOfTimes90DaysLate'] = 0
    dat.loc[dat['NumberOfTime60-89DaysPastDueNotWorse'].isin([96, 98]), 'NumberOfTime60-89DaysPastDueNotWorse'] = 0
    if(args.rs != None):
        random.seed(args.rs)

    kf = KFold(n_splits=5, shuffle=True, random_state=int(args.rs))

    for train_index, test_index in kf.split(range(len(dat))):
        # simple logistic regression model
        X_train = dat.iloc[train_index, 1:].values
        y_train = dat.iloc[train_index, 0].values
        X_test = dat.iloc[test_index, 1:].values
        y_test = dat.iloc[test_index, 0].values


        lr = LogisticRegression(solver='lbfgs')
        lr.fit(X_train, y_train)
        dat_dis = dat.copy()
        dat_dis['proba'] = lr.predict_proba(dat.iloc[:, 1:].values)[:, 1]
        dat_dis = dat_dis.loc[:, ['RevolvingUtilizationOfUnsecuredLines', 'age',
                          'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
                          'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
                          'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
                          'NumberOfDependents', 'proba', 'SeriousDlqin2yrs']]


        for var in ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'MonthlyIncome',
                    'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines']:

            dat_dis.loc[dat[var] <= dat[var]*0.2, var] = 0
            dat_dis.loc[(dat[var] <= dat[var]*0.4) & (dat[var] > dat[var]*0.2), var] = 1
            dat_dis.loc[(dat[var] <= dat[var]*0.6) & (dat[var] > dat[var]*0.4), var] = 2
            dat_dis.loc[(dat[var] <= dat[var]*0.8) & (dat[var] > dat[var]*0.6), var] = 3
            dat_dis.loc[(dat[var] > dat[var]*0.8), var] = 4


        dat_dis.loc[dat['age'] <= 20, 'age'] = 0
        dat_dis.loc[(dat['age'] > 20) & (dat['age'] <= 30), 'age'] = 1
        dat_dis.loc[(dat['age'] > 30) & (dat['age'] <= 40), 'age'] = 2
        dat_dis.loc[(dat['age'] > 40) & (dat['age'] <= 50), 'age'] = 3
        dat_dis.loc[(dat['age'] > 50) & (dat['age'] <= 60), 'age'] = 4
        dat_dis.loc[dat['age'] > 60, 'age'] = 5
        for var in ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'NumberOfDependents']:
            dat_dis.loc[dat[var] >= 5, var] = 5
        for var in ['NumberOfTime60-89DaysPastDueNotWorse']:
            dat_dis.loc[dat[var] >= 4, var] = 4


        df_prob_train = dat_dis.iloc[train_index, :].copy()
        df_prob_test = dat_dis.iloc[test_index, :].copy()
        bias_detection(df_prob_train, df_prob_test)