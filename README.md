## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd 
df= pd.read_csv("Encoding Data.csv")
df
```
<img width="276" height="337" alt="Screenshot 2025-10-16 220009" src="https://github.com/user-attachments/assets/0877390e-362f-4b46-9ce7-d232438b5f08" />

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm= ['Hot','Warm','Cold']
e1= OrdinalEncoder (categories=[pm])
e1.fit_transform (df[["ord_2"]])
```
<img width="189" height="210" alt="Screenshot 2025-10-16 220014" src="https://github.com/user-attachments/assets/7829c5c9-46b1-4d6c-b76e-a6c029cfc7ce" />

```
df['bo2']= e1.fit_transform(df[["ord_2"]])
df
```
<img width="329" height="335" alt="Screenshot 2025-10-16 220019" src="https://github.com/user-attachments/assets/a4d46057-48a6-47c2-8ee8-abf1039c58a8" />

```
le= LabelEncoder()
dfc= df.copy()
dfc['ord_2']=le.fit_transform (dfc['ord_2'])
dfc
```
<img width="341" height="350" alt="Screenshot 2025-10-16 220024" src="https://github.com/user-attachments/assets/4c6fd51a-5b1e-42a5-b4ef-8876b6fc9331" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2= pd.concat([df2,enc],axis=1)
df2
```
<img width="425" height="335" alt="Screenshot 2025-10-16 220030" src="https://github.com/user-attachments/assets/848fbda7-d2dd-4a50-a238-de42a3dcf919" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="648" height="351" alt="Screenshot 2025-10-16 220036" src="https://github.com/user-attachments/assets/237e71e8-f88b-4194-b821-5e850466ffce" />

```
from category_encoders import BinaryEncoder
df= pd.read_csv("data.csv")
df
```
<img width="455" height="342" alt="Screenshot 2025-10-16 220101" src="https://github.com/user-attachments/assets/73d9c2a9-2356-4583-8047-3edb49f99364" />

```
be= BinaryEncoder()
nd= be.fit_transform(df['Ord_2'])
df
```
<img width="480" height="347" alt="Screenshot 2025-10-16 220106" src="https://github.com/user-attachments/assets/f103a46c-c525-4ef0-aace-b8e82f248ced" />

```
dfb= pd.concat([df,nd],axis=1)
dfb
```
<img width="674" height="348" alt="Screenshot 2025-10-16 220111" src="https://github.com/user-attachments/assets/f7669f85-18ce-4075-b502-e063478fa657" />

```
from category_encoders import TargetEncoder
te= TargetEncoder()
CC= df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC= pd.concat([CC,new],axis=1)
CC
```
<img width="532" height="341" alt="Screenshot 2025-10-16 220115" src="https://github.com/user-attachments/assets/44d5d588-2037-45dd-a4e7-15216f6a2f01" />

```
import pandas as pd 
import numpy as np
from scipy import stats 
df= pd.read_csv("Data_to_Transform.csv")
df
```
<img width="719" height="400" alt="Screenshot 2025-10-16 220120" src="https://github.com/user-attachments/assets/bc5024a0-3cda-4805-94a2-e2786bd235cb" />

```
df.skew()
```
<img width="330" height="117" alt="Screenshot 2025-10-16 220126" src="https://github.com/user-attachments/assets/ad42ced3-1f26-4516-96ad-f504246103c4" />

```
np.log(df["Highly Positive Skew"])
```
<img width="533" height="250" alt="Screenshot 2025-10-16 220129" src="https://github.com/user-attachments/assets/61fa46d2-3c55-47a1-938a-577a3bd6611f" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="574" height="247" alt="Screenshot 2025-10-16 220135" src="https://github.com/user-attachments/assets/687b25d8-71e6-4b92-b1a0-9091d6fdacca" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="561" height="242" alt="Screenshot 2025-10-16 220139" src="https://github.com/user-attachments/assets/9abcf8b0-56c1-446c-9eee-68f9797fe797" />

```
np.square(df["Highly Positive Skew"])
```
<img width="538" height="253" alt="Screenshot 2025-10-16 220145" src="https://github.com/user-attachments/assets/e47e28df-73ac-4e41-834e-97bf3cbd0f54" />

```
df["Highly Positive Skew_boxcox"],parameters= stats.boxcox(df["Highly Positive Skew"])
df
```
<img width="924" height="408" alt="Screenshot 2025-10-16 220151" src="https://github.com/user-attachments/assets/b999eb3e-7f32-488a-9cf2-53a6c90927ff" />

```
df.skew()
```
<img width="379" height="140" alt="Screenshot 2025-10-16 220157" src="https://github.com/user-attachments/assets/963341a3-c02f-4e60-b9e4-3beb642e7c15" />

```
df["Highly Negative Skew_yeojohnson"],parameters= stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```
<img width="413" height="154" alt="Screenshot 2025-10-16 220205" src="https://github.com/user-attachments/assets/4716c0a4-290d-4be9-92e7-772d2df8e1d5" />

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal')
 df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 df
```
<img width="1154" height="429" alt="Screenshot 2025-10-16 220214" src="https://github.com/user-attachments/assets/e4b03348-603b-41d2-ae8c-7009df8ca35d" />

```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
<img width="699" height="503" alt="Screenshot 2025-10-16 220220" src="https://github.com/user-attachments/assets/36416d1f-ef14-462d-bbf9-9def4e450d8a" />

```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
<img width="690" height="489" alt="Screenshot 2025-10-16 220226" src="https://github.com/user-attachments/assets/6c66e60f-a6ee-41f0-82d7-948ec8efc3cb" />

```
 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
```
<img width="690" height="496" alt="Screenshot 2025-10-16 220232" src="https://github.com/user-attachments/assets/434050e6-3468-40cc-ab19-ab9f465b1679" />

```
 df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
 sm.qqplot(df["Highly Negative Skew"],line='45')
 plt.show()
```
<img width="706" height="498" alt="Screenshot 2025-10-16 220238" src="https://github.com/user-attachments/assets/6324b3e1-a9ec-49f3-8fde-2148aa24e00a" />

```
dt =pd.read_csv("titanic_dataset.csv")
dt
dt=pd.read_csv("titanic_dataset.csv")
dt
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
```
<img width="702" height="499" alt="Screenshot 2025-10-16 220243" src="https://github.com/user-attachments/assets/66a78383-ec65-491f-95d0-b8913358ea06" />

```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
<img width="704" height="497" alt="Screenshot 2025-10-16 220248" src="https://github.com/user-attachments/assets/ff0aa7cb-7b3f-47d5-a6be-47feb369a77e" />

# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file  was performed successfully


       
