import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import boxcox

# import statsmodels.api as sm

bodyfat = pd.read_csv("C:/Machine_learning/LinearregressionProject/bodyfat.csv")

# # print null values
print(bodyfat.isnull().sum())

# Store the 'BodyFat' column in a separate variable
body_fat = bodyfat['BodyFat']

# Remove the 'BodyFat' column from the original DataFrame
features = bodyfat.drop('BodyFat', axis=1)

# # Print the two variables

# print(body_fat)
# print(features)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(features, body_fat)

# Make predictions using the testing set
body_fat_predict = regr.predict(features)


# The coefficients
print("Coefficients: \n", regr.coef_)

# The intercept
print("Intercept: \n", regr.intercept_)

# create a vector of bodyfat predictions


## Testing ##
# pred=regr.predict([[1.0853,22,173.25,72.25,38.5,93.6,83.0,98.7,58.7,37.3,23.4,30.5,28.9,18.2]])
# print("Predicted body fat percentage: \n", pred)
# # Success

# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(body_fat, body_fat_predict))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(body_fat, body_fat_predict))



# plt.figure(figsize=(8,6))

# # Create a scatter plot with a red line indicating perfect predictions
# plt.scatter(body_fat, body_fat_predict, s=80, alpha=0.7, color="darkblue")
# plt.plot([min(body_fat), max(body_fat)], [min(body_fat), max(body_fat)], 'r--', linewidth=2)

# # Add axis labels, a title, a legend, and gridlines
# plt.xlabel('True value', fontsize=14)
# plt.ylabel('Predicted value', fontsize=14)
# plt.title('True vs Predicted', fontsize=16)
# # plt.legend(['Perfect prediction', 'Actual data'], fontsize=12, loc='lower right')
# plt.grid(alpha=0.3)

# # Show the plot
# plt.show()


# # Plot the residuals
# plt.figure(figsize=(8,6))

# # Create a scatter plot with blue points and reduced transparency
# plt.scatter(body_fat_predict, body_fat_predict - body_fat, c='steelblue', s=80, alpha=0.7)

# # Add a horizontal line at y=0 to indicate zero residuals
# plt.hlines(y=0, xmin=0, xmax=50, linestyles='dashed', linewidth=2, color='darkred')

# # Add axis labels and a title
# plt.xlabel('Predicted body fat', fontsize=14)
# plt.ylabel('Residuals', fontsize=14)
# plt.title('Residual plot', fontsize=16)

# # Show the plot
# plt.show()


# ### TASK 2 ###




m = mean_squared_error(body_fat, body_fat_predict)
d = len(features.columns) + 1

def cooks_distance(y_pred, y_pred_i, m, d):
    s = np.sum((y_pred - y_pred_i)**2)/(d*m)
    return s

cooks_distances = np.array([])
for i in range(len(features)):
    features_i = features.drop(i)
    body_fat_i = body_fat.drop(i)
    regr_i = linear_model.LinearRegression()
    regr_i.fit(features_i, body_fat_i)
    body_fat_pred_i = regr_i.predict(features)
    cd = cooks_distance(body_fat_predict, body_fat_pred_i, m, d)
    cooks_distances = np.append(cooks_distances, cd)

# np.set_printoptions(precision=10, suppress=True)
print("Cook's distance for each observation:\n", cooks_distances)


# # find outliers along with their index numbers
outliers = np.where(cooks_distances > 4/len(features))
print("Outlier values: ", body_fat[outliers[0]])
# print("Outliers: ", outliers)
# # print outlier values along with their index numbers



# remove outliers
features_withoutliers = features.drop(outliers[0])
body_fat_withoutliers = body_fat.drop(outliers[0])

# performing linear regression
regr_withoutliers = linear_model.LinearRegression()
regr_withoutliers.fit(features_withoutliers, body_fat_withoutliers)
body_fat_predict_withoutliers = regr_withoutliers.predict(features_withoutliers)

# The mean squared error
print("Mean squared error withou outliers: %.2f"% mean_squared_error(body_fat_withoutliers, body_fat_predict_withoutliers))

# Explained variance score: 1 is perfect prediction
print('Variance score without outliers: %.2f' % r2_score(body_fat_withoutliers, body_fat_predict_withoutliers))

# # Plot the true value vs the predicted values
# plt.scatter(body_fat_withoutliers, body_fat_predict_withoutliers,color="black")
# plt.xlabel('True value')
# plt.ylabel('Predicted value')
# plt.title('True vs Predicted')
# plt.show()

# # Plot the residuals
# plt.scatter(body_fat_predict_withoutliers, body_fat_predict_withoutliers - body_fat_withoutliers, c='b', s=40, alpha=0.5)
# plt.hlines(y=0, xmin=0, xmax=50)
# plt.title('Residual plot')
# plt.ylabel('Residuals')
# plt.show()




# # Plot Cook's distance
# plt.figure(figsize=(12,8))
# plt.stem(np.arange(len(cooks_distances)), cooks_distances, markerfmt=",")
# plt.xlabel("Observation number")
# plt.ylabel("Cook's distance")
# plt.title("Cook's distance plot")
# plt.show()

# ### TASK 3 ###


# Calculate the standardized residual for each data point. Plot the standardized residual vs the predicted values

# # Standardized residuals
standardized_residuals = (body_fat_withoutliers - body_fat_predict_withoutliers)/np.sqrt(m*(1 - r2_score(body_fat_withoutliers, body_fat_predict_withoutliers)))

print("Standard Residuals", np.array(standardized_residuals))

# # Plot the standardized residuals vs the predicted values
plt.scatter(body_fat_predict_withoutliers, standardized_residuals, color="black", alpha=0.5, s=50)
plt.xlabel('Predicted value')
plt.ylabel('Standardized residuals')
plt.title('Standardized residuals vs Predicted values')
plt.grid(True, linestyle='--', color='grey', alpha=0.5)
plt.gca().set_facecolor('whitesmoke')

# fit a linear regression line
slope, intercept = np.polyfit(body_fat_predict_withoutliers, standardized_residuals, 1)
x = np.linspace(min(body_fat_predict_withoutliers), max(body_fat_predict_withoutliers), 100)
y = slope*x + intercept

plt.plot(x, y, color='red')

plt.show()


# Do box cox transformation on bodyfat without outliers
body_fat_withoutliers_trans, lambda_val = boxcox(body_fat_withoutliers)

standardized_residuals1 = (body_fat_withoutliers_trans - body_fat_predict_withoutliers)/np.sqrt(m*(1 - r2_score(body_fat_withoutliers_trans, body_fat_predict_withoutliers)))

plt.scatter(standardized_residuals1, body_fat_predict_withoutliers, color="black", alpha=0.5, s=50)
plt.ylabel('Predicted value', fontsize=12)
plt.xlabel('Standardized residuals', fontsize=12)
plt.title('Standardized residuals vs Predicted values', fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', color='grey', alpha=0.5)
plt.gca().set_facecolor('whitesmoke')

# Fit a line to the plot
m, b = np.polyfit(standardized_residuals1, body_fat_predict_withoutliers, 1)
plt.plot(standardized_residuals1, m*standardized_residuals1 + b, color='red', linewidth=2)

plt.show()


print("Transformed data", body_fat_withoutliers_trans)
print("Lambda value", lambda_val)

### TASK-4 ####

regr_tran = linear_model.LinearRegression()

# Train the model using the training sets
regr_tran.fit(features_withoutliers, body_fat_withoutliers_trans)

# Make predictions using the testing set
body_fat_predict_new = regr_tran.predict(features_withoutliers)


# The coefficients
print("Coefficients: \n", regr_tran.coef_)

# The intercept
print("Intercept: \n", regr_tran.intercept_)

# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(body_fat, body_fat_predict))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(body_fat, body_fat_predict))


# Fit a line to the data
slope, intercept = np.polyfit(body_fat_withoutliers_trans, body_fat_predict_new, 1)

# Plot the scatter plot
plt.scatter(body_fat_withoutliers_trans, body_fat_predict_new)
plt.xlabel('True Values', fontsize=14)
plt.ylabel('Predictions', fontsize=14)
plt.title('True Values vs Predicted Values (Transformed)', fontsize=16)

# Plot the line of best fit
plt.plot([min(body_fat_withoutliers_trans), max(body_fat_withoutliers_trans)], [min(body_fat_withoutliers_trans), max(body_fat_withoutliers_trans)], 'r--', linewidth=2, label='Line of Best Fit')

# Add a legend and gridlines
plt.legend(fontsize=12)
plt.grid(True)

# Adjust the plot size and font sizes
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['font.size'] = 12

# Display the plot
plt.show()

