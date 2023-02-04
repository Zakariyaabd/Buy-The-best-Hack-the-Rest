"""
Create merged and cleaned data using the order and planned
data from FY21 and FY22.
"""
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


DATE_COL = "Delivery Week"
BAD_DATE_COL = "Distribution Date"
COST_COL = "Cost"
BAD_COST_COL = "Unit Cost"
BAD_PRICE_COL = "Unit Price"
PRICE_COL = "Price"

def merge_expected_actual(actual_path="FY21 LFM Order Items.csv",
                            expected_path = "FY21 Planning Items.csv"):
    actual = pd.read_csv(actual_path)
    expected = pd.read_csv(expected_path)

    expected[DATE_COL] = pd.to_datetime(expected[DATE_COL])
    expected.set_index(DATE_COL, inplace=True)
    actual.rename(columns={BAD_DATE_COL: DATE_COL,
                                BAD_COST_COL: COST_COL,
                                BAD_PRICE_COL: PRICE_COL},
                    inplace=True)
    actual[DATE_COL] = pd.to_datetime(actual[DATE_COL])
    actual.set_index(DATE_COL, inplace=True)

    actual[COST_COL] = actual[COST_COL].str.replace('$', '').astype(float)
    expected[COST_COL] = expected[COST_COL].str.replace('$', '').astype(float)
    actual[PRICE_COL] = actual[PRICE_COL].str.replace('$', '').astype(float)

    ts_grouper = pd.Grouper(freq='W-MON')
    actual = actual.groupby(["Producer Code", "Product ID", ts_grouper],
                        ).agg(sum).reset_index()
    expected = expected.groupby(["Producer Code", "Product ID", ts_grouper]
                        ).agg(sum).reset_index()

    # y is expected, x is actual
    merged_expected_actual = actual.merge(expected,
                            on=["Producer Code","Product ID","Delivery Week"])
    return merged_expected_actual


df_21 = merge_expected_actual()
df_22 = merge_expected_actual(actual_path="FY22_LFM_Order_Items_Updated.csv",
                    expected_path="FY22_Planning_Items_Updated.csv")


df = pd.concat([df_21,df_22])
del df_21, df_22

lwa = pd.read_csv("Land_Water Access.csv")

df = df.merge(lwa, on="Producer Code")
del lwa

df.rename(columns={"Quantity_x": "qty_actual",
    "Cost_x": "cost_actual",
    "Quantity_y": "qty_expected",
    "Cost_y": "cost_expected"}, inplace=True)


weather = pd.read_csv("weather.csv")

weather["Date"] = pd.to_datetime(weather["Date"])
weather.set_index("Date", inplace=True)

for col in weather.columns:
    weather.loc[weather[col]=="T",col] = 0
    weather[col] = weather[col].astype(float)

weather_agg_map = {'Maximum Temperature degrees (F)': "mean",
                    'Minimum Temperature degrees (F)': "mean",
                    'Precipitation (inches)': "mean",
                    'Snow (inches)': "mean",
                    'Snow Depth (inches)': sum}

weather = weather.resample("W-MON").agg(weather_agg_map)


weather = weather.rolling(4).agg(weather_agg_map)
df["Delivery Week"] = pd.to_datetime(df["Delivery Week"])


df = df.merge(weather.reset_index(),
                right_on="Date",
                left_on="Delivery Week",
                how="left")

df.drop(columns=["Date"], inplace=True)

df.to_csv("merged_data.csv", index=False)
