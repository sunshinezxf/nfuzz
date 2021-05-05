import numpy as np
import pandas as pd
data=pd.read_table(r'C:\Users\MI\Downloads\I-1-5(100).dat',sep='\s+',names=['x','y','z'])
print(data)
# data = np.array([[1, 2, 3], [1, 5, 6], [7, 8, 9]])
df = pd.DataFrame(data)
#给df加上列名
df.columns = ['x','y','z']
#要被group的列名
cols = ['x','y','z']
#根据哪个列进行group
gp_col = 'x'
#df_mean中存储group后的结果
df_mean = df.groupby(gp_col)[cols].mean()
print(df_mean)
