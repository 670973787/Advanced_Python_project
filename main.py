if __name__ == "__main__":

    args = parser.parse_args()

    print('Args:', args)
    PEL_RATE_ORI = 5
    MAX_ITER = 100
    csv_file_vr = args.data
    dat = pd.read_csv(csv_file_vr, index_col=0)
    dat = dat[:1500]
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
    

        
        