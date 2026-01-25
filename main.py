import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
pt_x=PowerTransformer(method="yeo-johnson")

X_train=pt_x.fit_transform(x_train)
X_test=pt_x.transform(x_test)

pt_y=PowerTransformer(method="yeo-johnson")
Y_train_resize=y_train.values.reshape(-1,1)
Y_test_resize=y_test.values.reshape(-1,1)

Y_train_tf=pt_y.fit_transform(Y_train_resize)
Y_test_tf=pt_y.transform(Y_test_resize)

#plt.figure(figsize=(6,4))
#plt.scatter(X_train[:, 1], Y_train_tf)
#plt.xlabel("Age (Transformed)")
#plt.ylabel("Charges (Transformed)")
#plt.title("Age vs Charges (After Transformation)")
#plt.show()


lr=LinearRegression()
X_train_lr=lr.fit(X_train,Y_train_tf)

Y_pre=lr.predict(X_test)

print("R2score",r2_score(Y_test_tf,Y_pre))


bmi=int(input(" Enter your  Body Mass Index :"))
age=int(input("enter your age:"))
smoker=input("do you smoke  yes or No :").lower()

if smoker=="yes":
    smoker=1
else:
    smoker=0
list1=pd.DataFrame(
    [[bmi,age,smoker]],
    columns=["bmi","age","smoker"])

list_1=pt_x.transform(list1)

y_predict=lr.predict(list_1)

list_2=pt_y.inverse_transform(y_predict)

print("your medical insurance is:",list_2)