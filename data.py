import pandas as pd 

df = pd.read_csv("merged_with_unique_prod.csv")

grouped = df.groupby("Unique Product").agg({"qty_actual": "sum", "qty_expected": "sum"})
grouped["probability_delivery"] = grouped["qty_actual"] / grouped["qty_expected"]

df["probability_delivery"] = df["Unique Product"].apply(lambda x: grouped.loc[x, "probability_delivery"])

df.to_csv("new_mergedData.csv", index=False)


df = pd.read_csv("new_mergedData.csv")

# Sort the DataFrame in descending order based on the "probability_delivery" column
df.sort_values(by="probability_delivery", ascending=False, inplace=True)

# Select the top 3 rows of the sorted DataFrame
top3 = df.head(3)
botton3 = df.tail(3)

result = [top3, botton3]

print(result)


