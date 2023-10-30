# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from MLPipeline.Featurizing import Featurizing
from MLPipeline.MLP import MLP

# Set the default Seaborn style
seaborn.set()

# Import data from an Excel file
raw_csv_data = pd.read_excel("../Input/CallCenterData.xlsx")

# Create a copy of the original data for comparison
df_comp = raw_csv_data.copy()

# Plot and save visualizations for different columns in the dataset
df_comp.Healthcare.plot(figsize=(20, 5), title="Healthcare")
plt.savefig("../Output/plots/" + "healthcare.png")

df_comp.Telecom.plot(figsize=(20, 5), title="Telecom")
plt.savefig("../Output/plots/" + "Telecom.png")

df_comp.Banking.plot(figsize=(20, 5), title="Banking")
plt.savefig("../Output/plots/" + "Banking.png")

df_comp.Technology.plot(figsize=(20, 5), title="Technology")
plt.savefig("../Output/plots/" + "Technology.png")

df_comp.Insurance.plot(figsize=(20, 5), title="Insurance")
plt.savefig("../Output/plots/" + "Insurance.png")

# Define a new dataframe with required features
df = df_comp[["month", "Healthcare"]]


'''
# define a function for cesium model
def cesium_exec(df):

    df['ts'] = df['month'].apply(lambda x: x.timestamp()).astype(int)
    ### Reshaping the data
    n_past = 4
    target_data = []
    for i in range(len(df)):
        temp = []
        time = []
        for j in range(n_past + 1):
            try:
                temp.append(df.Healthcare[i + j])
                time.append(df.ts[i + j])
            except Exception as e:
                continue
        if len(temp) > 4:
            try:
                target_data.append([np.array(temp), np.array(time), df.Healthcare[i + j + 1]])
            except Exception as e:
                continue

    cesium_df = pd.DataFrame(target_data).rename(columns={0: 'y', 1: 'ts', 2: "target"})

    cs_df = cesium_df[['ts', 'y']].to_dict('list')

    # introducing the feature
    fset_cesium = Featurizing().exec(cs_df)

    fset_cesium["target"] = cesium_df["target"]

    X = fset_cesium.drop("target", axis=1).values
    Y = fset_cesium["target"]

    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    MLP(X_train, Y_train, X_test, Y_test)
'''


# Define a function for fbprophet model
def fbprophet_exec(df):
    # Rename columns for compatibility with fbprophet
    df = df.rename(columns={'month': 'ds', 'Healthcare': 'y'})

    # Create a time series plot
    ax = df.set_index('ds').plot(figsize=(20, 8)
    ax.set_ylabel('Monthly Number of Healthcare Queries')
    ax.set_xlabel('Date')

    # Save the plot to an output folder
    plt.savefig("../Output/plots/" + "fbprophet.png")

    # Execute the fbprophet predictions using the imported model
    Prophet(df)


# run the cesium model
# cesium_exec(df)

# OR

# Run the fbprophet model
fbprophet_exec(df)

