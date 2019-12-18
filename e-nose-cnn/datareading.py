import pandas as pd
import numpy as np

train_x = pd.read_csv( "new.csv")
# train_y = pd.read_excel( "F:\\gitworkspace\\python\\e-nose-cnn\\codedata\\3times\\label.xlsx")

# test_x = pd.read_csv( "F:\\gitworkspace\\python\\e-nose-cnn\\nos-data\\3times7class\\3\\testset2.csv")
# test_y = pd.read_excel( "F:\\gitworkspace\\python\\e-nose-cnn\\nos-data\\3times7class\\test_label2.xlsx")

train_x = train_x.fillna(method='ffill')
# test_x = test_x.fillna(method='ffill')

np.save('new.npy', train_x)
print(train_x)
# np.save('F:\\gitworkspace\\python\\e-nose-cnn\\nos-data\\3times7class\\trainlabel2.npy', train_y)

# np.save('F:\\gitworkspace\\python\\e-nose-cnn\\nos-data\\3times7class\\3\\testset2.npy', test_x)
# np.save('F:\\gitworkspace\\python\\e-nose-cnn\\nos-data\\3times7class\\testlabel2.npy', test_y)