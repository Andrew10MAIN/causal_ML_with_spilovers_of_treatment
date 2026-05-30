import numpy as np
import pandas as pd
#import geopandas as gpd
#from shapely.geometry import Point
#from libpysal.weights import DistanceBand
#from scipy.sparse import identity
#from scipy.sparse.linalg import spsolve
#from scipy.spatial import cKDTree

import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from causalml.inference.meta import BaseSRegressor
from econml.dml import CausalForestDML
from sklearn.multioutput import MultiOutputRegressor

def make_treatment_effects_df(df_arg, rings_list, model_suffix, treated_col='treated', #real_col='ITE_real'
                              ):
    results = []

    for ring in rings_list:

        series = df_arg[df_arg[treated_col] == ring][ring]

        att = series.mean()

        se = series.std(ddof=1) / np.sqrt(len(series))

        ci_low = att - 1.96 * se
        ci_high = att + 1.96 * se

        results.append({
            'ring': ring,
            f'att_{model_suffix}': att,
            f'se_{model_suffix}': se,
            #'ci_low': ci_low,
            #'ci_high': ci_high,
            #'ITE_real_mean': ite_real_mean,
            #'n': len(series)
        })

    return pd.DataFrame(results)


def process_single_key(i, replacing_dict_0_ring, replacing_dict_odr_ring, dict_of_gdfs, dict_of_combs):

    # =========================================================
    # 0. Data preparation
    # =========================================================

    temp_gdf = dict_of_gdfs[i].copy()

    df_effectbase = pd.DataFrame({
        'ring': [
            'treated_inner_ring',
            'treated_outer_ring1',
            'treated_outer_ring2',
            'treated_outer_ring3',
            'treated_outer_ring4'
        ],
        'true_effect': [
            temp_gdf[temp_gdf['T'] == 1]['tau'].mean(),
            temp_gdf[temp_gdf['ODR_1'] == 1]['tau'].mean(),
            temp_gdf[temp_gdf['ODR_2'] == 1]['tau'].mean(),
            temp_gdf[temp_gdf['ODR_3'] == 1]['tau'].mean(),
            temp_gdf[temp_gdf['ODR_4'] == 1]['tau'].mean(),
        ],
    })

    X = temp_gdf[["C1","C2"]].values

    T_M = temp_gdf['T_tot_cat'].astype(str).values

    Y = temp_gdf['Y_dep_var_ns'].values

    T_mult = temp_gdf[
        ['T', 'ODR_1','ODR_2','ODR_3','ODR_4']
    ].to_numpy()

    T_D_id = temp_gdf[
        ['T', 'Cont_T_N']
    ].to_numpy()

    Xdf_did = temp_gdf[
        ['T', 'ODR_1','ODR_2','ODR_3','ODR_4',"C1","C2"]
    ]

    X_did = sm.add_constant(Xdf_did)

    y_did = temp_gdf['Y_dep_var_ns']

    # =========================================================
    # 1. Multi-treatment S-Learner
    # =========================================================

    mS_learn = BaseSRegressor(
        RandomForestRegressor(
            n_estimators=160,
            max_depth=10,
            random_state=42,
            n_jobs=1
        ),
        control_name='control'
    )

    ite_mS_learn = mS_learn.fit_predict(X, T_M, Y)

    ITE_df_mSlearn = pd.DataFrame(ite_mS_learn).rename(columns={
        0:'treated_inner_ring',
        1:'treated_outer_ring1',
        2:'treated_outer_ring2',
        3:'treated_outer_ring3',
        4:'treated_outer_ring4',
    })

    ITE_df_mSlearn['treated'] = T_M

    df_multi_treatment_effects_Slearn = make_treatment_effects_df(
        ITE_df_mSlearn,
        [
            'treated_inner_ring',
            'treated_outer_ring1',
            'treated_outer_ring2',
            'treated_outer_ring3',
            'treated_outer_ring4',
        ],
        'mSlearn',
        treated_col='treated'
    )

    df_multi_effect_pre1 = df_effectbase.merge(
        df_multi_treatment_effects_Slearn,
        on='ring',
        how='left'
    )

    # =========================================================
    # 2. Multi-treatment Causal Forest
    # =========================================================

    mCF = CausalForestDML(
        model_y=RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=20,
            n_jobs=1
        ),
        model_t=MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=20,
                n_jobs=1
            )
        ),
        cv=None,
        criterion='mse',
        n_estimators=1000,
        min_samples_leaf=10,
        min_impurity_decrease=0.001,
        random_state=123
    )

    mCF.tune(Y, T_mult, X=X)
    mCF.fit(Y, T_mult, X=X)

    mCF_pred = mCF.const_marginal_effect(X)

    ITE_mCF = pd.DataFrame(mCF_pred).rename(
        columns=replacing_dict_0_ring
    )

    ITE_mCF['treated'] = T_M

    df_multi_treatment_effects_mCF = make_treatment_effects_df(
        ITE_mCF,
        [
            'treated_inner_ring',
            'treated_outer_ring1',
            'treated_outer_ring2',
            'treated_outer_ring3',
            'treated_outer_ring4'
        ],
        'mCF',
        treated_col='treated'
    )

    df_multi_effect_pre2 = df_multi_effect_pre1.merge(
        df_multi_treatment_effects_mCF,
        on='ring',
        how='left'
    )

    # =========================================================
    # 3. Double-treatment Causal Forest
    # =========================================================

    dCF = CausalForestDML(
        model_y=RandomForestRegressor(
            n_estimators=100,
            max_depth=3,
            min_samples_leaf=20,
            n_jobs=1
        ),
        model_t=MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=100,
                max_depth=3,
                min_samples_leaf=20,
                n_jobs=1
            )
        ),
        cv=None,
        criterion='mse',
        n_estimators=1000,
        min_samples_leaf=10,
        min_impurity_decrease=0.001,
        random_state=123
    )

    dCF.tune(Y, T_D_id, X=X)
    dCF.fit(Y, T_D_id, X=X)

    dCF_pred = dCF.const_marginal_effect(X)

    ITE_dCF = pd.DataFrame(dCF_pred).rename(columns={
        0:'treated_inner_ring',
        1:'treated_outer_rings',
    })

    ITE_dCF['treated'] = T_M

    ITE_dCF['treated_outer_ring1'] = ITE_dCF['treated_outer_rings']
    ITE_dCF['treated_outer_ring2'] = ITE_dCF['treated_outer_rings']/2
    ITE_dCF['treated_outer_ring3'] = ITE_dCF['treated_outer_rings']/3
    ITE_dCF['treated_outer_ring4'] = ITE_dCF['treated_outer_rings']/4

    df_multi_treatment_effects_dCF = make_treatment_effects_df(
        ITE_dCF,
        [
            'treated_inner_ring',
            'treated_outer_ring1',
            'treated_outer_ring2',
            'treated_outer_ring3',
            'treated_outer_ring4'
        ],
        'dCF',
        treated_col='treated'
    )

    df_multi_effect_pre3 = df_multi_effect_pre2.merge(
        df_multi_treatment_effects_dCF,
        on='ring',
        how='left'
    )

    # =========================================================
    # 4. Difference in Differences
    # =========================================================

    model_did = sm.OLS(y_did, X_did)
    results_did = model_did.fit()

    coef_df_did = pd.DataFrame({
        "ring": results_did.params.index,
        "att_mDiD": results_did.params.values,
        "se_mDiD": results_did.bse.values
    })

    coef_df_did = coef_df_did[
        coef_df_did['ring'].isin([
            'T', 'ODR_1', 'ODR_2', 'ODR_3', 'ODR_4'
        ])
    ].copy()

    coef_df_did['ring'] = coef_df_did['ring'].replace(
        replacing_dict_odr_ring
    )

    df_multi_effect_temp = df_multi_effect_pre3.merge(
        coef_df_did,
        on='ring',
        how='left'
    )

    df_multi_effect_temp['effect_size'] = (
        1 / dict_of_combs[i][2] * 100
    )

    df_multi_effect_temp = df_multi_effect_temp[
        [
            'effect_size',
            'ring',
            'true_effect',
            'att_mSlearn',
            'se_mSlearn',
            'att_mCF',
            'se_mCF',
            'att_dCF',
            'se_dCF',
            'att_mDiD',
            'se_mDiD'
        ]
    ]

    return df_multi_effect_temp