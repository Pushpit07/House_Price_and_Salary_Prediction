import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

bias = 100
X, y, coef = make_regression(n_features=1, bias=bias, noise=10, random_state=42, coef=True)

print(coef, ',', bias)

# plt.scatter(X, y)
# plt.show()

y_gen = (coef * X) + bias

# plt.scatter(X, y)
# plt.plot(X, y_gen)
# plt.show()

# lets plot a line using linear regression
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

print(model.coef_,',', model.intercept_)

# plt.scatter(X, y)
# plt.plot(X, y_gen, label="Pre")
# plt.plot(X, model.predict(X), label="LR")
# plt.legend()
# plt.show()



class LinearRegCustom:
    # constructor
    def __init__(self, lr=0.1):
        self.lr = lr
    
    # training function i.e. fit
    def fit(self, X, y):
        self._X = X # _X mock behavior like private
        self._y = y.reshape(-1, 1) # do calculations, else it will give error due to some numpy shape
        
        # need to figure out value of coef & intercept
        # step 1: pick these values at random 
        self.coef_ = np.random.random()
        self.intercept_ = np.random.random()
        
        # gradient descent
        errors = []
        
        # lets say we do this 50 times
        for i in range(50):
            self.gradient_decend()
            errors.append(self.error())
        return errors
    
    def gradient_decend(self):
        # change in coef and intercept
        d_coef, d_intercept = self.gradient()
        self.coef_ -= d_coef * self.lr
        self.intercept_ -= d_intercept * self.lr
    
    def gradient(self):
        yh = self.predict(self._X) # from predict funtion
        
        d_coef = ((yh - self._y) * self._X).mean()
        d_intercept = (yh - self._y).mean()
        
        return d_coef, d_intercept
    
    def predict(self, X):
        return X * self.coef_ + self.intercept_
    
    def error(self):
        return ((self.predict(self._X) - self._y) ** 2).sum()


model = LinearRegCustom(lr=0.1)

errors = model.fit(X, y)

print(model.coef_, ',', model.intercept_)

# plt.scatter(X, y)
# plt.plot(X, y_gen, label="Pre")
# plt.plot(X, model.predict(X), label="CLR")
# plt.legend()
# plt.show()

# plt.plot(errors)
# plt.show()


for i in range(1, 10):
    model = LinearRegCustom(lr=.5 * i)
    errors = model.fit(X, y)
    
    plt.figure()
    plt.title(str(.5 * i))
    plt.plot(errors)

plt.show()