"""
Script for computing fulfillment probabilities for various products/producers.
"""
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("merged_data.csv")

df["Owns Land"] = df["Owns Land"].astype(int)
df["Access to Water"] = df["Access to Water"].astype(int)
df["fullfilled"] = (df["qty_actual"]>=df["qty_expected"]).astype(int)

product_wise = df[["Product ID","fullfilled"]].groupby("Product ID",
                            as_index=False).mean()
product_time_wise =  df[["Product ID","Delivery Week","fullfilled"]
                        ].groupby(["Product ID","Delivery Week"],
                            as_index=False).mean()
producer_time_wise = df[["Producer Code","Delivery Week","fullfilled"]
                        ].groupby(["Producer Code","Delivery Week"],
                            as_index=False).mean()
producer_wise = df[["Producer Code","fullfilled"]].groupby("Producer Code",
                            as_index=False).mean()

plt.style.use("ggplot")

def plot_shiz(data, x, y="fullfilled",  top_n=10, ascending=True):
    data[x] = data[x].astype(str)
    data["mask"] = data[y]>=0.99
    data["mask"] = data["mask"].map({True: "green", False: "grey"})
    data = data.sort_values(y, ascending=ascending)[:top_n]
    plt.figure(figsize=(15,6))
    plt.bar(data[x], data[y], color=data["mask"])
    plt.xlabel(x)
    plt.ylabel("Probability of Fulfilment")


plot_shiz(producer_wise, x="Producer Code", top_n=20, ascending=False)

plot_shiz(product_wise, x="Product ID", top_n=30, ascending=False)
