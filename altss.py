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
    while (cur_best_score != best_score) and num_iter <= 100:
        num_iter += 1

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
