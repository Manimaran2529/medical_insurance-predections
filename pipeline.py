import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
mm=pd.read_csv("medical_insurance.csv")
mm=mm.drop_duplicates()

mm["sex"]=mm["sex"].map({"female":1,"male":0})

mm["smoker"]=mm["smoker"].map({"yes":1,"no":0})

mm=pd.get_dummies(mm ,columns=["region"] ,dtype=int)

#cor=mm.corr()
#plt.figure(figsize=(10,10))
#sns.heatmap(cor,annot=True,cmap="coolwarm")
#plt.show()

x=mm[["bmi","age","smoker"]]
y=mm["charges"]
 
#plt.subplot(1,2,1)
#ns.histplot(mm["age"])
#plt.title("Age Distribution")

#plt.subplot(1,2,2)
#sns.histplot(mm["bmi"])
#plt.title("BMI Distribution")

#plt.show()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)
 
pipe=Pipeline([
    ("PowerTransformer",PowerTransformer(method="yeo-johnson")),
    ("LinearRegression",LinearRegression())
])
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)

print("r2_score",r2_score(y_test,y_pred))