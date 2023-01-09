#%%
from __init__ import get_dataset
import numpy as np
#%%
# test1
data = get_dataset('Synthetic', 'MMoE_Synthetic/syn_0.9.csv')
assert data is not None
# %%
# npdata = data.numerical_data

# %%
# correcoef = np.corrcoef(npdata)
# %%
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.subplots(figsize=(9, 9))
# sns.heatmap(correcoef, annot=True, vmax=1, square=True, cmap="Blues")
# plt.show()

# %%
# test2
data = get_dataset('AliExpress', 'AliExpress_NL/test.csv')
assert data is not None
len(data)
# %%
