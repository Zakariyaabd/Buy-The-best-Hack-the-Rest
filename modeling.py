"""
Script imports processed and cleaned data and fits a linear regression
model to infer the correlation of target (fulfillment) vs the
various external factors and categories (product wise or producer wise)
"""
import numpy as np
import pandas as pd
import statsmodels.api as sma


df = pd.read_csv("merged_data.csv")

df["Owns Land"] = df["Owns Land"].astype(int)
df["Access to Water"] = df["Access to Water"].astype(int)

agg_map = {'qty_actual': sum,
            'cost_actual': np.mean,
            'Price': np.mean,
            'qty_expected': sum,
            'cost_expected': np.mean,
            'Owns Land': np.mean,
            'Access to Water': np.mean}

weather_agg_map = {**agg_map, **{'Maximum Temperature degrees (F)': "mean",
                                'Minimum Temperature degrees (F)': "mean",
                                'Precipitation (inches)': sum,
                                'Snow (inches)': sum,
                                'Snow Depth (inches)': "mean"}}

product_wise = df.groupby("Product ID",
                            as_index=False).agg(agg_map)
product_time_wise =  df.groupby(["Product ID","Delivery Week"],
                            as_index=False).agg(weather_agg_map)
producer_time_wise = df.groupby(["Producer Code","Delivery Week"],
                            as_index=False).agg(weather_agg_map)
producer_wise = df.groupby("Producer Code",
                            as_index=False).agg(agg_map)


df["fullfilled"] = (df["qty_actual"]>=df["qty_expected"]).astype(int)
product_wise["fullfilled"] = (product_wise["qty_actual"]>=product_wise["qty_expected"]).astype(int)
product_time_wise["fullfilled"] = (
    product_time_wise["qty_actual"]>=product_time_wise["qty_expected"]
                                ).astype(int)
producer_wise["fullfilled"] = (
    producer_wise["qty_actual"]>=producer_wise["qty_expected"]
                                ).astype(int)
producer_time_wise["fullfilled"] = (
    producer_time_wise["qty_actual"]>=producer_time_wise["qty_expected"]
                                ).astype(int)


def perform_linear_regression_analysis(reg_data: pd.DataFrame,
                    target_col: str,
                    features: list[str],
                    fe_group_col: list[str]=[""],
                    fixed_effects: bool=False):
    """Perform linear regression with y=target column and features as the X
    (dependent variables). We can also perform the fixed effects regression
    to see effects of each category in `fe_group_col`.

    Args:
        reg_data ([pandas.DataFrame])
        target_col (str)
        features (list[str])
        fe_group_col (list[str], optional)
        fixed_effects (bool, optional)

    Returns:
        [type]: [description]
    """
    reg_y = reg_data[target_col]

    if fixed_effects:
        reg_data = pd.get_dummies(reg_data, columns=fe_group_col)
        features += [col for col in reg_data.columns
                    if (col not in features) and col.startswith(fe_group_col+"_")]

    if "constant" not in reg_data.columns:
        reg_data["constant"] = 1
        features.append("constant")

    reg = sma.OLS(reg_y, reg_data[features], hasconst=True)
    model = reg.fit()

    col_array = np.array(features)
    col_array = col_array[np.argsort(model.pvalues)][::-1]

    feat_imp = dict(zip(col_array, np.sort(model.pvalues)[::-1]))

    predictor_coeffs = dict(zip(features, model.params))
    print(model.summary())
    return feat_imp, predictor_coeffs

feat_cols = ['Price', 'qty_expected',
            'cost_expected', 'Owns Land',
            'Access to Water']
weather_feat_cols = feat_cols +['Maximum Temperature degrees (F)',
                    'Snow Depth (inches)',
                    'Snow (inches)',
                    'Precipitation (inches)',
                    'Minimum Temperature degrees (F)']

_ = perform_linear_regression_analysis(df, "fullfilled", weather_feat_cols.copy())


new_agg_map = weather_agg_map.copy()
new_agg_map["fullfilled"] = "mean"
producer_wise = df.groupby("Producer Code",
                            as_index=False).agg(new_agg_map)

_ = perform_linear_regression_analysis(producer_wise, "fullfilled", weather_feat_cols)


_ = perform_linear_regression_analysis(producer_time_wise, "fullfilled",
                                            features=feat_cols.copy(),
                                            fe_group_col="Producer Code",
                                            fixed_effects=True)


_ = perform_linear_regression_analysis(product_time_wise,
                                        "fullfilled",
                                        feat_cols.copy(),
                                        fe_group_col="Product ID",
                                        fixed_effects=True)


_ = perform_linear_regression_analysis(product_time_wise, "fullfilled", weather_feat_cols)
