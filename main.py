import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib


#-----------------------------------------------------------------------------------------------------------------
#1 Лінійна регресія з 1 змінною
#-----------------------------------------------------------------------------------------------------------------

dataset = pd.read_csv('kc_house_data.csv')
dataset_describe = dataset.describe()
print(dataset_describe)
print(dataset.info())
print(dataset.shape)
print(dataset.columns)

scatter_plot_sqft_price = dataset.plot.scatter(x='sqft_living', y='price',
                                               title='Relationship between sqft_living and price')
graph = scatter_plot_sqft_price.get_figure()
graph.savefig('sqft_price_relationship.png')

x = np.array(dataset[['sqft_living']]).reshape((-1, 1))
y = np.array(dataset['price'])
model = LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print(model.score(x_test, y_test))
joblib.dump(model, 'linear_regression_model.pkl')
#-----------------------------------------------------------------------------------------------------------------
# 2 Лінійна регресія з багатьма змінними
#-----------------------------------------------------------------------------------------------------------------

y_2 = np.asarray(dataset['price'].values.tolist())
y_2 = y_2.reshape(len(y_2), 1)
x_2 = np.array(dataset.drop(['price', 'date'], axis=1))
print(y_2.shape, x_2.shape)

scaler = StandardScaler()
x_scaler = scaler.fit_transform(x_2)
x_difference = x_2.max() - x_2.min()
x_scaler_difference = x_scaler.max() - x_scaler.min()
print(f'Difference x {x_difference} Difference after scaler {x_scaler_difference} ')

x_2_train, x_2_test, y_2_train, y_2_test = train_test_split(x_scaler, y_2, test_size=0.2, random_state=10)
model_2 = LinearRegression()
model_2.fit(x_2_train, y_2_train)
y_2_pred = model_2.predict(x_2_test)
mse_2 = mean_squared_error(y_2_test, y_2_pred)
print("Mean Squared Error:", mse_2)
joblib.dump(model_2, 'linear_regression_model2.pkl')
print(model_2.score(x_2_test, y_2_test))

x_train_normal_eq = np.hstack((np.ones((x_2_train.shape[0], 1)), x_2_train))
x_test_normal_eq = np.hstack((np.ones((x_2_test.shape[0], 1)), x_2_test))

theta = np.linalg.inv(x_train_normal_eq.T @ x_train_normal_eq) @ x_train_normal_eq.T @ y_2_train
y_pred_normal_eq = x_test_normal_eq @ theta
mse_normal_eq = mean_squared_error(y_2_test, y_pred_normal_eq)
print("Mean Squared Error (Normal Equation):", mse_normal_eq)

if mse < mse_2 and mse < mse_normal_eq:
    print("Model 1 is the best")
elif mse_2 < mse and mse_2 < mse_normal_eq:
    print("Model 2 is the best")
else:
    print("Model 3 is the best")

plt.figure(figsize=(10, 6))
plt.scatter(x_test, y_test, color='black', label='Actual Price')
plt.scatter(x_test, y_pred, color='blue', label='Linear Regression (1 Variable)')
plt.scatter(x_test, y_2_pred, color='red', label='Linear Regression (Multiple Variables)')
plt.scatter(x_test, y_pred_normal_eq, color='green', label='Linear Regression (Normal Equation)')
plt.xlabel('Sqft Living')
plt.ylabel('Price')
plt.title('Comparison of Models')
plt.legend()
plt.savefig('model_comparison.png')
plt.show()

#-----------------------------------------------------------------------------------------------------------------
# 3 Написати функцію
#-----------------------------------------------------------------------------------------------------------------


def loss_function(y: np.ndarray, x: np.ndarray) -> float:

    """"
    Calculates the loss function between the vectors of labels and predicted values

    Parameters:
    y (np.ndarray): The vector of true labels or target values
    x (np.ndarray): The vector of predicted values obtained from the model

    Returns:
    float: The value of the loss function

    """

    squared_errors = np.square(y - x)
    loss = np.mean(squared_errors)
    return loss

y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.2, 2.9, 4.1, 4.9])
result = loss_function(y_true, y_pred)
print(result)


