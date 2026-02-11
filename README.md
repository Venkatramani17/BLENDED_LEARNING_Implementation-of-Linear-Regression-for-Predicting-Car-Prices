# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries: Bring in essential libraries such as pandas, numpy, matplotlib, and sklearn.
2.Load Dataset: Import the dataset containing car prices along with relevant features.
3.Data Preprocessing: Manage missing data and select key features for the model, if required.
4.Split Data: Divide the dataset into training and testing subsets.
5.Train Model: Build a linear regression model and train it using the training data.
6.Make Predictions: Apply the model to predict outcomes for the test set.
7.Evaluate Model: Measure the model's performance using metrics like R² score, Mean Absolute Error (MAE), etc.
8.Check Assumptions: Plot residuals to verify assumptions like homoscedasticity, normality, and linearity.
9.Output Results: Present the predictions and evaluation metrics.

## Program:
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: R VENKATRAMANI
RegisterNumber:  25010118
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df=pd.read_csv('CarPrice_Assignment.csv')

print(df.head())

X = df[['horsepower','enginesize', 'citympg', 'highwaympg']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print("Name: R Venkatramani ")
print("Reg No: 212225240182 ")
print("\n\nMODEL COEFFICIENTS : \n")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")
print(f"{'Intercept'}: {model.intercept_}")

print("\n\nMODEL PERFORMANCE : \n")
print(f"{'MSE'}: {mean_squared_error(y_test,y_pred)}")

print(f"{'RMSE'}: {np.sqrt(mean_squared_error(y_test,y_pred))}")

print(f"{'MAE'}: {mean_absolute_error(y_test,y_pred)}")

print(f"{'R-squared'}: {r2_score(y_test,y_pred)}")

plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price ($)")
plt.grid(True)
plt.show()

residuals= y_test - y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"\nDurbin-Watson Statistics: {dw_test:.2f}",
      "\n(Values close to 2 indicate no autocorrelation)")

plt.figure(figsize=(10,5))
sns.residplot(x=y_pred, y=residuals, lowess=True, line_kws={'color':'red'})
plt.title("Homoscedasticity Check: Residuals vs Predicted")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Residuals ($)")
plt.grid(True)
plt.show()

fig, (ax1,ax2)=plt.subplots(1,2, figsize=(12,5))
sns.histplot(residuals,kde=True,ax=ax1)
ax1.set_title("Residuals Distribution")
sm.qqplot(residuals,line='45',fit=True,ax=ax2)
ax2.set_title("Q-Q Plot")
plt.tight_layout()
plt.show()
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
<img width="1055" height="326" alt="image" src="https://github.com/user-attachments/assets/8b74d7f9-e184-4d97-ac88-0efc188c47e1" />
<img width="1002" height="501" alt="image" src="https://github.com/user-attachments/assets/2c8d7839-55b4-40a3-9be7-7a263ed75e7f" />
<img width="997" height="544" alt="image" src="https://github.com/user-attachments/assets/19259cbe-d16a-43ef-a4d5-6409ebe73a4b" />
<img width="1068" height="423" alt="image" src="https://github.com/user-attachments/assets/22b0bad2-48e8-414d-a18d-19ab0951191e" />



## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
