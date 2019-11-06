import os
import sys
import pandas as pd
import numpy as np

df = pd.read_csv("F:\\gitworkspace\\python\\e-nose-cnn\\draw_test\\draw_test.csv")
df.values
# data = df.as_matrix()
data = df.values
data = list(map(list,zip(*data)))
data = pd.DataFrame(data)
data.to_csv(os.path.join("F:\\gitworkspace\\python\\e-nose-cnn\\draw_test", "tran.csv"), encoding='utf_8_sig', header=0, index=0)
print(data)
