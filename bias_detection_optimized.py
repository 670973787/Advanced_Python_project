#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%%writefile project.py

from mpi4py import MPI

def bias_detection(dat, dat_test):
    global PEL_RATE

    PEL_RATE = PEL_RATE_ORI * len(dat) / 7214
    
    rank = MPI.COMM_WORLD.Get_rank()
    
    if rank == 0: 
        _, _, feature_filter_right = bias_subscan(dat, "right")

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
        print("under train = {} with p value {}".format(train_score_under,p_right))
        _, test_score_under = unpel_score(dat_test, feature_filter_right, mode='right')
        print("estimated under test ={}".format(test_score_under))
        PEL_RATE = 0
        _, _, test_filt_right = bias_subscan(dat_test, "right")
        _, test_score_under = unpel_score(dat_test, test_filt_right, mode='right')
        print("true under test ={}".format(test_score_under))
       
        
    if rank == 1:
        _, _, feature_filter_left = bias_subscan(dat, "left")
        
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

        print("over train = {} with p_value {}".format(train_score_over,p_left))
        
        _, test_score_over = unpel_score(dat_test, feature_filter_left, mode='left')
        print("estimated over test ={}".format(test_score_over))
        
        
        _, _, test_filt_left = bias_subscan(dat_test, "left")
        PEL_RATE = 0
        _, _, test_filt_left = bias_subscan(dat_test, "left")
        _, test_score_over = unpel_score(dat_test, test_filt_left, mode='left')
        print("true over test ={}".format(test_score_over))
     
    return None


# In[ ]:




