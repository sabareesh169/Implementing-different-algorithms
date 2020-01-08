
import numpy as np
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
from sklearn.model_selection import KFold
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import SGDRegressor
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings("ignore")

# Read in the training data and do intial preprocessing of dropping a seperate ID row and splitting SegmentID and time.
train_data=pd.read_csv("training_data.csv")
train_data[['SegmentId','Time']]=train_data["SegmentId_Time"].str.split('_',expand=True)
train_data.drop(columns=['Id','SegmentId_Time'],axis=1,inplace=True)

# We process the test_data in the same way as the training data
test_data=pd.read_csv("flight_testFeatures_toPredict.csv")
test_data[['SegmentId','Time']]=test_data["SegmentId_Time"].str.split('_',expand=True)
test_data.drop(columns=['Id','SegmentId_Time'],axis=1,inplace=True)

# Dropping SegmentID and time to test 
X_train=train_data.drop(['SegmentId','Time','Altitude','Latitude','Longitude'],axis=1)
Y_train=pd.DataFrame(train_data[['Altitude','Latitude','Longitude']])
X_test=test_data.drop(['SegmentId','Time','Altitude','Latitude','Longitude'],axis=1)

X_train_1, X_val, Y_train_1, Y_val = train_test_split(X_train, Y_train, test_size=0.4)

X_train_1.head(n=5)

Y_train.head(n=5)

fig = plt.figure(figsize=(24,6))
ax1 = fig.add_subplot(1,3,1)
ax1.plot(np.array(X_train)[:,0])
ax1.set_xlabel('$observation$')
ax1.set_ylabel('$Altitude$')
ax1.set_title('Atitude')
plt.show()

fig = plt.figure(figsize=(24,6))
ax1 = fig.add_subplot(1,3,1)
ax1.plot(np.arange(1000), np.array(X_train_1)[:,0][:1000])
ax1.set_xlabel('$observation$')
ax1.set_ylabel('$Altitude$')
ax1.set_title('Atitude')
plt.show()

print('Altitude Statistics: \n',stats.describe(np.array(X_train)[:,0]))
print('\nLatitude Statistics:\n',stats.describe(np.array(X_train)[:,1]))
print('\nLongitude Statistics:\n',stats.describe(np.array(X_train)[:,2]))

fig = plt.figure(figsize=(24,6))
ax1 = fig.add_subplot(1,3,1)
ax1.plot(np.array(X_train)[:,0])
ax1.set_xlabel('$observation$')
ax1.set_ylabel('$Altitude$')
ax1.set_title('Atitude')
ax2 = fig.add_subplot(1,3,2)
ax2.plot(np.array(X_train)[:,1])
ax2.set_xlabel('$observation$')
ax2.set_ylabel('$Latitude$')
ax2.set_title('Latitude')
ax3 = fig.add_subplot(1,3,3)
ax3.plot(np.array(X_train)[:,2])
ax3.set_xlabel('$observation$')
ax3.set_ylabel('$Longitude$')
ax3.set_title('Longitude')
plt.show()


fig = plt.figure(figsize=(24,6))
ax1 = fig.add_subplot(1,3,1)
ax1.plot(np.array(X_train)[:,0],label='Noisy')
ax1.legend()
ax1.plot(np.array(Y_train)[:,0],label='Target')
ax1.legend()
ax1.set_xlabel('$observation$')
ax1.set_ylabel('$Altitude$')
ax1.set_title('Noisy data vs Target for Atitude')
ax2 = fig.add_subplot(1,3,2)
ax2.plot(np.array(X_train)[:,1],label='Noisy')
ax2.legend()
ax2.plot(np.array(Y_train)[:,1],label='Target')
ax2.legend()
ax2.set_xlabel('$observation$')
ax2.set_ylabel('$Latitude$')
ax2.set_title('Noisy data vs Target for Latitude')
ax3 = fig.add_subplot(1,3,3)
ax3.plot(np.array(X_train)[:,2],label='Noisy')
ax3.legend()
ax3.plot(np.array(Y_train)[:,2],label='Target')
ax3.legend()
ax3.set_xlabel('$observation$')
ax3.set_ylabel('$Longitude$')
ax3.set_title('Noisy data vs Target for Longitude')
plt.show()


fig = plt.figure(figsize=(24,6))
ax1 = fig.add_subplot(1,3,1)
ax1.plot(np.arange(1000), np.array(X_train)[:1000,0],label='Noisy')
ax1.legend()
ax1.plot(np.arange(1000), np.array(Y_train)[:1000,0],label='Target')
ax1.legend()
ax1.set_xlabel('$observation$')
ax1.set_ylabel('$Altitude$')
ax1.set_title('Noisy data vs Target for Atitude')
ax2 = fig.add_subplot(1,3,2)
ax2.plot(np.arange(1000), np.array(X_train)[:1000,1],label='Noisy')
ax2.legend()
ax2.plot(np.arange(1000), np.array(Y_train)[:1000,1],label='Target')
ax2.legend()
ax2.set_xlabel('$observation$')
ax2.set_ylabel('$Latitude$')
ax2.set_title('Noisy data vs Target for Latitude')
ax3 = fig.add_subplot(1,3,3)
ax3.plot(np.arange(1000), np.array(X_train)[:1000,2],label='Noisy')
ax3.legend()
ax3.plot(np.arange(1000), np.array(Y_train)[:1000,2],label='Target')
ax3.legend()
ax3.set_xlabel('$observation$')
ax3.set_ylabel('$Longitude$')
ax3.set_title('Noisy data vs Target for Longitude')
plt.show()

scaler=StandardScaler().fit(X_train_1)

X_train_1_c=scaler.transform(X_train_1)
X_val_c=scaler.transform(X_val)
clf = SGDRegressor(loss='huber',alpha=0)

size=np.array([0.01, 0.1, 1, 50, 100])
i=0

pred_altitude_test=np.zeros(5)
pred_altitude_train=np.zeros(5)
pred_latitude_test=np.zeros(5)
pred_latitude_train=np.zeros(5)
pred_longitude_test=np.zeros(5)
pred_longitude_train=np.zeros(5)

clf = SGDRegressor(loss='huber',alpha=0)

indices = list(range(0,X_train_1.shape[0]))
np.random.shuffle(indices)
train_data_2 = np.array(X_train_1_c)[indices,:]
train_y_2 = np.array(Y_train_1)[indices,:]
for j in size:
    train_data_1 = train_data_2[:int(j*train_data_2.shape[0]/100),:]
    train_y_1 = train_y_2[:int(j*train_y_2.shape[0]/100),:]
    
    clf = SGDRegressor(loss='huber',alpha=0, n_iter=100)
    train_latitude=np.hstack([np.ones((train_data_1.shape[0],1)),np.array(train_data_1)[:,1].reshape(-1,1)])
    clf.fit(train_latitude, np.array(train_y_1)[:,1])
    test_latitude=np.hstack([np.ones((X_val.shape[0],1)),np.array(X_val_c)[:,1].reshape(-1,1)])
    pred_latitude_ = clf.predict(test_latitude)
    pred_latitude_test[i] = mean_squared_error(pred_latitude_, np.array(Y_val)[:,1])
    pred_latitude_train[i] = mean_squared_error(clf.predict(train_latitude), np.array(train_y_1)[:,1])

    train_longitude=np.hstack([np.ones((train_data_1.shape[0],1)),np.array(train_data_1)[:,2].reshape(-1,1)])
    clf.fit(train_longitude, np.array(train_y_1)[:,2])
    test_longitude=np.hstack([np.ones((X_val.shape[0],1)),np.array(X_val_c)[:,2].reshape(-1,1)])
    pred_longitude_ = clf.predict(test_longitude)
    pred_longitude_test[i] = mean_squared_error(pred_longitude_, np.array(Y_val)[:,2])
    pred_longitude_train[i] = mean_squared_error(clf.predict(train_longitude), np.array(train_y_1)[:,2])
    
    clf = SGDRegressor(loss='huber',alpha=0, n_iter=700)
    train_altitude=np.hstack([np.ones((train_data_1.shape[0],1)),np.array(train_data_1)[:,0].reshape(-1,1)])
    clf.fit(train_altitude, np.array(train_y_1)[:,0])
    test_altitude=np.hstack([np.ones((X_val.shape[0],1)),np.array(X_val_c)[:,0].reshape(-1,1)])
    pred_altitude_ = clf.predict(test_altitude)
    pred_altitude_test[i] = mean_squared_error(pred_altitude_, np.array(Y_val)[:,0])
    pred_altitude_train[i] = mean_squared_error(clf.predict(train_altitude), np.array(train_y_1)[:,0])
    
    print(i)
    i = i+1


SGD_accuracy = pred_altitude_test+pred_latitude_test+pred_longitude_test
SGD_accuracy


fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(np.log(size/100),(pred_altitude_test+pred_latitude_test+pred_longitude_test)/3,label='Testing error')
ax1.legend()
ax1.plot(np.log(size/100), (pred_altitude_train+pred_latitude_train+pred_longitude_train)/3, label='Training Accuracy')
ax1.legend()
ax1.set_xlabel('percentage of training dataset (log scale)')
ax1.set_ylabel('$Error$')
ax1.set_title('Learning curves')
plt.show()

clf = SGDRegressor(loss='huber',alpha=0)
train_latitude=np.hstack([np.ones((X_train_1.shape[0],1)),np.array(X_train_1_c)[:,1].reshape(-1,1)])
clf.fit(train_latitude, np.array(Y_train_1)[:,1])
test_latitude=np.hstack([np.ones((X_val.shape[0],1)),np.array(X_val_c)[:,1].reshape(-1,1)])
pred_latitude = clf.predict(test_latitude)

train_longitude=np.hstack([np.ones((X_train_1.shape[0],1)),np.array(X_train_1_c)[:,2].reshape(-1,1)])
clf.fit(train_longitude, np.array(Y_train_1)[:,2])
test_longitude=np.hstack([np.ones((X_val.shape[0],1)),np.array(X_val_c)[:,2].reshape(-1,1)])
pred_longitude = clf.predict(test_longitude)

clf = SGDRegressor(loss='huber',alpha=0, n_iter=1000)
train_altitude=np.hstack([np.ones((X_train_1.shape[0],1)),np.array(X_train_1_c)[:,0].reshape(-1,1)])
clf.fit(train_altitude, np.array(Y_train_1)[:,0])
test_altitude=np.hstack([np.ones((X_val.shape[0],1)),np.array(X_val_c)[:,0].reshape(-1,1)])
pred_altitude = clf.predict(test_altitude)


fig = plt.figure(figsize=(24,6))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(pred_altitude[:1000],label='Prediction')
ax1.legend()
ax1.plot(np.array(Y_val)[:1000,0],label='Target')
ax1.legend()
ax1.set_xlabel('$observation$')
ax1.set_ylabel('$Altitude$')
ax1.set_title('Altitude')
plt.show()

fig = plt.figure(figsize=(24,6))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(pred_latitude[:1000],label='Prediction')
ax1.legend()
ax1.plot(np.array(Y_val)[:1000,1],label='Target')
ax1.legend()
ax1.set_xlabel('$observation$')
ax1.set_ylabel('$Latitude$')
ax1.set_title('Latitude')
plt.show()

fig = plt.figure(figsize=(24,6))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(pred_longitude[:1000],label='Prediction')
ax1.legend()
ax1.plot(np.array(Y_val)[:1000,2],label='Target')
ax1.legend()
ax1.set_xlabel('$observation$')
ax1.set_ylabel('$Longitude$')
ax1.set_title('Longitude')
plt.show()


kf = KFold(n_splits=10)
SGD_regressor=np.zeros((10,3))
i=0
for train, test in kf.split(X_train):
    train_data = np.array(X_train)[train]
    train_y = np.array(Y_train)[train]
    test_data = np.array(X_train)[test]
    test_y = np.array(Y_train)[test]
    scaler_SGD = StandardScaler().fit(train_data)
    train_data = scaler_SGD.transform(train_data)
    test_data = scaler_SGD.transform(test_data)
    
    clf = SGDRegressor(loss='huber',alpha=0, n_iter=100)
    train_latitude=np.hstack([np.ones((train_data.shape[0],1)),np.array(train_data)[:,1].reshape(-1,1)])
    clf.fit(train_latitude, np.array(train_y)[:,1])
    test_latitude=np.hstack([np.ones((test_data.shape[0],1)),np.array(test_data)[:,1].reshape(-1,1)])
    pred_latitude = clf.predict(test_latitude)
    SGD_regressor[i,1]=mean_squared_error(pred_latitude, np.array(test_y)[:,1])


    train_longitude=np.hstack([np.ones((train_data.shape[0],1)),np.array(train_data)[:,2].reshape(-1,1)])
    clf.fit(train_longitude, np.array(train_y)[:,1])
    test_longitude=np.hstack([np.ones((test_data.shape[0],1)),np.array(test_data)[:,2].reshape(-1,1)])
    pred_longitude = clf.predict(test_longitude)
    SGD_regressor[i,2]=mean_squared_error(pred_longitude, np.array(test_y)[:,2])
    
    clf = SGDRegressor(loss='huber',alpha=0, n_iter=700)
    train_altitude=np.hstack([np.ones((train_data.shape[0],1)),np.array(train_data)[:,0].reshape(-1,1)])
    clf.fit(train_altitude, np.array(train_y)[:,0])
    test_altitude=np.hstack([np.ones((test_data.shape[0],1)),np.array(test_data)[:,0].reshape(-1,1)])
    pred_altitude = clf.predict(test_altitude)
    SGD_regressor[i,0]=mean_squared_error(pred_altitude, np.array(test_y)[:,0])
    
    i = i+1


params = {'max_leaf_nodes': [2, 5, 10], 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeRegressor(random_state=42), params)
grid_search_cv.fit(X_val,Y_val)
grid_search_cv.best_estimator_


params = {'max_leaf_nodes': [10, 20, 30], 'max_depth': [10, 20]}
grid_search_cv = GridSearchCV(DecisionTreeRegressor(random_state=42), params)
grid_search_cv.fit(X_val,Y_val)
grid_search_cv.best_estimator_


params = {'max_leaf_nodes': [30, 40, 50], 'max_depth': [5, 10]}
grid_search_cv = GridSearchCV(DecisionTreeRegressor(random_state=42), params)
grid_search_cv.fit(X_val,Y_val)
grid_search_cv.best_estimator_


params = {'max_leaf_nodes': [50, 60, 70]}
grid_search_cv = GridSearchCV(DecisionTreeRegressor(random_state=42), params)
grid_search_cv.fit(X_val,Y_val)
grid_search_cv.best_estimator_


params = {'max_leaf_nodes': [70, 100]}
grid_search_cv = GridSearchCV(DecisionTreeRegressor(random_state=42), params)
grid_search_cv.fit(X_val,Y_val)
grid_search_cv.best_estimator_


clf_dec = DecisionTreeRegressor(max_depth=10, max_leaf_nodes=100, min_samples_split=2)
clf_dec.fit(np.array(X_train_1)[:,0].reshape(-1,1), np.array(Y_train_1)[:,0])
pred_altitude=clf_dec.predict(np.array(X_val)[:,0].reshape(-1,1))
clf_dec.fit(np.array(X_train_1)[:,1].reshape(-1,1), np.array(Y_train_1)[:,1])
pred_latitude=clf_dec.predict(np.array(X_val)[:,1].reshape(-1,1))
clf_dec.fit(np.array(X_train_1)[:,2].reshape(-1,1), np.array(Y_train_1)[:,2])
pred_longitude=clf_dec.predict(np.array(X_val)[:,2].reshape(-1,1))


fig = plt.figure(figsize=(24,6))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(pred_altitude[:1000],label='Prediction')
ax1.legend()
ax1.plot(np.array(Y_val)[:1000,0],label='Target')
ax1.legend()
ax1.set_xlabel('$observation$')
ax1.set_ylabel('$Altitude$')
ax1.set_title('Altitude')
plt.show()

fig = plt.figure(figsize=(24,6))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(pred_latitude[:1000],label='Prediction')
ax1.legend()
ax1.plot(np.array(Y_val)[:1000,1],label='Target')
ax1.legend()
ax1.set_xlabel('$observation$')
ax1.set_ylabel('$Latitude$')
ax1.set_title('Latitude')
plt.show()

fig = plt.figure(figsize=(24,6))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(pred_longitude[:1000],label='Prediction')
ax1.legend()
ax1.plot(np.array(Y_val)[:1000,2],label='Target')
ax1.legend()
ax1.set_xlabel('$observation$')
ax1.set_ylabel('$Longitude$')
ax1.set_title('Longitude')
plt.show()


size=np.array([.1, 1, 5, 50, 100])
i=0

pred_altitude_test=np.zeros(5)
pred_altitude_train=np.zeros(5)
pred_latitude_test=np.zeros(5)
pred_latitude_train=np.zeros(5)
pred_longitude_test=np.zeros(5)
pred_longitude_train=np.zeros(5)

clf_dec = DecisionTreeRegressor(max_depth=100, max_leaf_nodes=500, min_samples_split=2)

indices = list(range(0,X_train_1.shape[0]))
np.random.shuffle(indices)
train_data_2 = np.array(X_train_1_c)[indices,:]
train_y_2 = np.array(Y_train_1)[indices,:]
for j in size:
    train_data_1 = train_data_2[:int(j*train_data_2.shape[0]/100),:]
    train_y_1 = train_y_2[:int(j*train_y_2.shape[0]/100),:]
    
    train_latitude = np.array(train_data_1)[:,1].reshape(-1,1)
    clf_dec.fit(train_latitude, np.array(train_y_1)[:,1])
    test_latitude=np.array(X_val_c)[:,1].reshape(-1,1)
    pred_latitude_test[i] = mean_squared_error(clf_dec.predict(test_latitude), np.array(Y_val)[:,1])
    pred_latitude_train[i] = mean_squared_error(clf_dec.predict(train_latitude), np.array(train_y_1)[:,1])

    train_longitude=np.array(train_data_1)[:,2].reshape(-1,1)
    clf_dec.fit(train_longitude, np.array(train_y_1)[:,2])
    test_longitude=np.array(X_val_c)[:,2].reshape(-1,1)
    pred_longitude_test[i] = mean_squared_error(clf_dec.predict(test_longitude), np.array(Y_val)[:,2])
    pred_longitude_train[i] = mean_squared_error(clf_dec.predict(train_longitude), np.array(train_y_1)[:,2])
    
    train_altitude=np.array(train_data_1)[:,0].reshape(-1,1)
    clf_dec.fit(train_altitude, np.array(train_y_1)[:,0])
    test_altitude=np.array(X_val_c)[:,0].reshape(-1,1)
    pred_altitude_test[i] = mean_squared_error(clf_dec.predict(test_altitude), np.array(Y_val)[:,0])
    pred_altitude_train[i] = mean_squared_error(clf_dec.predict(train_altitude), np.array(train_y_1)[:,0])
    i = i+1

fig = plt.figure(figsize=(8,6))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(np.log(size/100),(pred_altitude_test+pred_latitude_test+pred_longitude_test)/3,label='Testing Error')
ax1.legend()
ax1.plot(np.log(size/100), (pred_altitude_train+pred_latitude_train+pred_longitude_train)/3, label='Training Error')
ax1.legend()
ax1.set_xlabel('percentage of training dataset (log scale)')
ax1.set_ylabel('$Error$')
ax1.set_title('Learning curves')
plt.show()


i=0
Dec_tree=np.zeros((10,3))
clf_dec = DecisionTreeRegressor(max_depth=100, max_leaf_nodes=500, min_samples_split=2)

for train, test in kf.split(X_train):
    train_data = np.array(X_train)[train]
    train_y = np.array(Y_train)[train]
    test_data = np.array(X_train)[test]
    test_y = np.array(Y_train)[test]

    clf_dec.fit(train_data[:,0].reshape(-1,1), train_y[:,0].reshape(-1,1))
    pred_altitude = clf_dec.predict(test_data[:,0].reshape(-1,1))
    Dec_tree[i,0]=mean_squared_error(pred_altitude, test_y[:,0].reshape(-1,1))

    clf_dec.fit(train_data[:,1].reshape(-1,1), train_y[:,1].reshape(-1,1))
    pred_latitude = clf_dec.predict(test_data[:,1].reshape(-1,1))
    Dec_tree[i,1]=mean_squared_error(pred_latitude, test_y[:,1].reshape(-1,1))


    clf_dec.fit(train_data[:,2].reshape(-1,1), train_y[:,2].reshape(-1,1))
    pred_altitude = clf_dec.predict(test_data[:,2].reshape(-1,1))
    Dec_tree[i,2]=mean_squared_error(pred_altitude, test_y[:,2].reshape(-1,1))
    
    i = i+1


def bagged_tree(n, X, Y, x_test_1, model):
    pred_ = np.zeros((x_test_1.shape[0],3))
    for i in range(n):
        x_test = np.zeros((x_test_1.shape))
        a=np.random.randint(1,100)
        training_data=X.sample(200000, replace=True, random_state=a)
        training_y = Y.sample(200000, replace=True, random_state=a)
        scaler_1=StandardScaler().fit(np.array(training_data))

        x_test=np.array(scaler_1.transform(x_test_1))
        training_data=scaler_1.transform(training_data)
        training_y=np.array(training_y)
        
        model.fit(training_data[:,0].reshape(-1,1), training_y[:,0].reshape(-1,1))
        pred_altitude = model.predict(x_test[:,0].reshape(-1,1))
        
        model.fit(training_data[:,1].reshape(-1,1), training_y[:,1].reshape(-1,1))
        pred_latitude = model.predict(x_test[:,1].reshape(-1,1))
        
        model.fit(training_data[:,2].reshape(-1,1), training_y[:,2].reshape(-1,1))
        pred_longitude = model.predict(x_test[:,2].reshape(-1,1))
                
        pred_ += np.hstack([pred_altitude.reshape(-1,1), pred_latitude.reshape(-1,1), pred_longitude.reshape(-1,1)])        
    
    prediction = pred_/n
    return prediction


clf_dec = DecisionTreeRegressor(max_depth=100, max_leaf_nodes=500, min_samples_split=2)
bagged_scores=np.zeros(8)
for i in range(8):
    z=mean_squared_error(bagged_tree(10, X_train_1, Y_train_1, X_val, clf_dec),Y_val)
    bagged_scores[i]=z


accuracy=[bagged_scores, Dec_tree.sum(axis=1)/3 ]
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)    
bp = ax.boxplot(accuracy)
plt.xticks([1, 2, 3], ['Bagged Decision trees', 'Decision trees'])
ax.set_axisbelow(True)
ax.set_title('Comparison of different models')
ax.set_xlabel('Models')
ax.set_ylabel('Error')
plt.show()


difference=Dec_tree[:8].sum(axis=1)/3-bagged_scores
mean_difference=np.mean(difference)
sd_difference=np.std(difference)
se_mean_difference=sd_difference/np.sqrt(7)
t_stat=mean_difference/se_mean_difference
print('The obtained t-statistic value is :',t_stat)
print('The p value for this t-value is 0.0566 which isless than assumed alpha(0.05). Hence we cannot reject the null hypothesis.')

def create_model(idim, odim, hdim1, hdim2):
    model = torch.nn.Sequential(
            torch.nn.Linear(idim, hdim1),
            torch.nn.LeakyReLU(0.1),
        
            torch.nn.Linear(hdim1, hdim2),
            torch.nn.LeakyReLU(0.1),
        
            torch.nn.Linear(hdim2, odim)
            )
    return model
def nn_train(train_x, train_y, model, epoch, lrate,minibatch_size):
    inputs=[]

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
    for itr in range(epoch):
        indices = list(range(0,train_x.shape[0]))
        np.random.shuffle(indices)
        X_train = np.array(train_x)[indices,:]
        y_train = np.array(train_y)[indices,:]

        for i in range(0, X_train.shape[0], minibatch_size):
            # Get pair of (X, y) of the current mini-batch
            X_train_mini = X_train[i:i + minibatch_size]
            y_train_mini = y_train[i:i + minibatch_size]
            
            X = autograd.Variable(torch.from_numpy((X_train_mini).astype(float)), requires_grad=True).float()
            Y = autograd.Variable(torch.from_numpy(y_train_mini), requires_grad=False).float()
    
            y_pred = model(X)
            loss = loss_fn(y_pred, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

def nn_test(test_x, model):
    X = autograd.Variable(torch.from_numpy((test_x).astype(float)), requires_grad=False).float()
    y_pred = model(X)
    return y_pred

idim = 3  # input dimension
hdim1 = 32 # hidden layer one dimension
hdim2 = 50 # hidden layer two dimension
odim = 3   # output dimension

size=np.array([.1, 1, 5, 50, 100])

scaler=StandardScaler()

nn_training_score=np.zeros(5)
nn_testing_score=np.zeros(5)
i=0
indices = list(range(0,X_train_1.shape[0]))
np.random.shuffle(indices)
train_data_1 = np.array(X_train_1)[indices,:]
train_y_1 = np.array(Y_train_1)[indices,:]
for j in size:
    model = create_model(idim, odim, hdim1, hdim2, hdim3) # creating model structure
    train_data = train_data_1[:int(j*train_data_1.shape[0]/100),:]
    train_y = train_y_1[:int(j*train_y_1.shape[0]/100),:]
    test_data = np.array(X_val)
    test_y = np.array(Y_val)
    train_data = scaler.fit_transform(train_data)
    trained_model = nn_train(train_data, train_y, model, 250, 0.05, 2000) # training model
    test_data = scaler.transform(test_data)
    test=(nn_test(test_data, trained_model)).data.numpy()
    train=(nn_test(train_data, trained_model)).data.numpy()
    nn_testing_score[i]=mean_squared_error(test, test_y) # testing model
    nn_training_score[i]=mean_squared_error(train, train_y) # testing model
    i = i+1


indices = list(range(0,X_train_1.shape[0]))
np.random.shuffle(indices)
train_data_1 = np.array(X_train_1)[indices,:]
train_y_1 = np.array(Y_train_1)[indices,:]
scaler=StandardScaler().fit(train_data_1)
train_data_1=scaler.transform(train_data_1)
test_data = np.array(X_val)
test_data = scaler.transform(test_data)
model = create_model(idim, odim, hdim1, hdim2) # creating model structure
trained_model = nn_train(train_data_1, train_y_1, model, 500, 0.05, 500)
test_result = nn_test(test_data, trained_model)


prediction = test_result.data.numpy()
fig = plt.figure(figsize=(24,6))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(prediction[:1000,0],label='Prediction')
ax1.legend()
ax1.plot(np.array(Y_val)[:1000,0],label='Target')
ax1.legend()
ax1.set_xlabel('$observation$')
ax1.set_ylabel('$Altitude$')
ax1.set_title('Altitude')
plt.show()

fig = plt.figure(figsize=(24,6))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(prediction[:1000,1],label='Prediction')
ax1.legend()
ax1.plot(np.array(Y_val)[:1000,1],label='Target')
ax1.legend()
ax1.set_xlabel('$observation$')
ax1.set_ylabel('$Latitude$')
ax1.set_title('Latitude')
plt.show()

fig = plt.figure(figsize=(24,6))
ax1 = fig.add_subplot(1,1,1)
ax1.plot(prediction[:1000,2],label='Prediction')
ax1.legend()
ax1.plot(np.array(Y_val)[:1000,2],label='Target')
ax1.legend()
ax1.set_xlabel('$observation$')
ax1.set_ylabel('$Longitude$')
ax1.set_title('Longitude')
plt.show()


idim = 3  # input dimension
hdim1 = 32 # hidden layer one dimension
hdim2 = 50 # hidden layer two dimension
odim = 3   # output dimension
scaler=StandardScaler()
model = create_model(idim, odim, hdim1, hdim2) # creating model structure
kf = KFold(n_splits=10)

nn_testing_score=np.zeros(12)
i=0
for train, test in kf.split(X_train):
    train_data = np.array(X_train)[train]
    train_y = np.array(Y_train)[train]
    test_data = np.array(X_train)[test]
    test_y = np.array(Y_train)[test]
    
    train_data = scaler.fit_transform(train_data)
    trained_model = nn_train(train_data, train_y, model, 600, 0.1, 20000) # training model

    test_data = scaler.transform(test_data)
    test=(nn_test(test_data, trained_model)).data.numpy()
    nn_testing_score[i]=mean_squared_error(test, test_y) # testing model
    print(i)
    i = i+1

accuracy=[nn_testing_score_,SGD_accuracy/3,Dec_tree.sum(axis=1)/3 ]
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)    
bp = ax.boxplot(accuracy)
plt.xticks([1, 2, 3], ['Neural Network', 'Linear Regression', 'Decision trees'])
ax.set_axisbelow(True)
ax.set_title('Comparison of different models')
ax.set_xlabel('Models')
ax.set_ylabel('Error')
plt.show()


difference=Dec_tree[:8,:].sum(axis=1)/3-SGD_accuracy/3
mean_difference=np.mean(difference)
sd_difference=np.std(difference)
se_mean_difference=sd_difference/np.sqrt(7)
t_stat=mean_difference/se_mean_difference
print('The obtained t-statistic value is :',t_stat)


print('The p value for this t-value is 0.054 which is greater than the assumed alpha(0.05/3). Hence we cannot reject the null hypothesis.')


difference=nn_testing_score_-SGD_accuracy/3
mean_difference=np.mean(difference)
sd_difference=np.std(difference)
se_mean_difference=sd_difference/np.sqrt(7)
t_stat=mean_difference/se_mean_difference
print('The obtained t-statistic value is :',t_stat)


print('The p value for this t-value is 0.0356 which is greater than assumed alpha(0.05/3). Hence we cannot reject the null hypothesis.')


difference=nn_testing_score_-Dec_tree[:8,:].sum(axis=1)/3
mean_difference=np.mean(difference)
sd_difference=np.std(difference)
se_mean_difference=sd_difference/np.sqrt(7)
t_stat=mean_difference/se_mean_difference
print('The obtained t-statistic value is :',t_stat)


print('The p value for this t-value is 0.0862 which is greater than assumed alpha(0.05/3). Hence we cannot reject the null hypothesis.')


i=0
Dec_tree=np.zeros((10,3))
clf_dec = DecisionTreeRegressor(max_depth=100, max_leaf_nodes=500, min_samples_split=2)

for train, test in kf.split(X_train):
    train_data =X_train.iloc[train]
    train_y = Y_train.iloc[train]
    test_data = X_train.iloc[test]
    test_y = Y_train.iloc[test]

    clf_dec.fit(train_data[:,0].reshape(-1,1), train_y[:,0].reshape(-1,1))
    pred_altitude = bagged_tree(10, train_data[:,0], train_y[:0], test_data[:0], clf_dec)
    Dec_tree[i,0]=mean_squared_error(pred_altitude, test_y[:,0].reshape(-1,1))

    clf_dec.fit(train_data[:,1].reshape(-1,1), train_y[:,1].reshape(-1,1))
    pred_altitude = bagged_tree(10, train_data[:,1], train_y[:1], test_data[:1], clf_dec)
    Dec_tree[i,1]=mean_squared_error(pred_altitude, test_y[:,1].reshape(-1,1))


    clf_dec.fit(train_data[:,2].reshape(-1,1), train_y[:,2].reshape(-1,1))
    pred_altitude = bagged_tree(10, train_data[:,2], train_y[:2], test_data[:2], clf_dec)
    Dec_tree[i,2]=mean_squared_error(pred_altitude, test_y[:,2].reshape(-1,1))
    
    i = i+1
