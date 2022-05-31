# (Double check if I need all these)
import numpy as np

from matplotlib.pyplot import subplots

# Try this to disable warnings
import warnings
warnings.filterwarnings('ignore')

def norm_b(data):
    max_data = np.max(data)
    min_data = np.min(data)
    if abs(min_data) > max_data:
        max_data = abs(min_data)
    data = data / max_data
    return data

# %% [markdown]
# ### (0, 1)
# Check which normalization is the best workin w audio files

# %%
def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = 2*((data-min_data) / (max_data - min_data))-1
    return data

