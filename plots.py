# %%
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# %%
data = pd.read_csv("data/rings3-regular-test.csv")
dump = pd.read_csv("rings3")
sns.scatterplot(data.x, data.y, data.c)
plt.show()
sns.scatterplot(data.x, data.y, dump[dump.columns[0]])
plt.show()
# %%
data = pd.read_csv("data/easy-test.csv")
dump = pd.read_csv("easy")
sns.scatterplot(data.x, data.y, data.c)
plt.show()
sns.scatterplot(data.x, data.y, dump[dump.columns[0]])
plt.show()
# %%
data = pd.read_csv("data/xor3-test.csv")
dump = pd.read_csv("xor3")
sns.scatterplot(data.x, data.y, data.c)
plt.show()
sns.scatterplot(data.x, data.y, dump[dump.columns[0]])
plt.show()
# %%
