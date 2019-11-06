import os
import pandas as pd
import numpy as np

data = [0,1,2,3,4,5,6,7,8,9]
data = pd.DataFrame(data)
data = data.values
data = list(map(list,zip(*data)))
data = pd.DataFrame(data)
data.to_csv(os.path.join("F:\\gitworkspace\\python\\e-nose-cnn\\train\\all_csv", "test.csv"), encoding='utf_8_sig', index=False, header=0)