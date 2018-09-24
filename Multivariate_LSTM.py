import numpy as np
from pandas import read_excel, DataFrame, concat
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler

#Loading the raw dataset (ignore the first 2 columns A & B)
dataset = read_excel("challenge_dataset.xlsx", usecols="C:Q")

values = dataset.values

print(dataset.head(3)) # Lets print the first 3 rows of the dataset

Nc = values.shape[1] #number of columns

#Lets create a quick plot of each series (as a separate subplot) and see what we have
i = 0
pyplot.figure()
for group in range(0,Nc):
    i += 1
    pyplot.subplot(Nc, 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(dataset.columns[group], y=0.5, loc='right')
pyplot.show()


values = values.astype('float32') # ensuring all the data is float

# normalizing features (columns 1 to 14) NOTE: column 15 = goal (three classes: 0, 1, 2)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values[:,0:Nc])
#scaled = values[:,0:Nc-1] #without scaling

print()
print("values.shape =", values.shape)
print("scaled.shape =", scaled.shape)

##############################################################

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame the time series as a supervised learning dataset.
    
    Arguments:
        data:    Sequence of observations as a list or NumPy array.
        n_in:    Number of lag observations as input (X).
        n_out:   Number of observations as output (y).
        dropnan: Boolean, whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """

    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # future sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg

# frame as supervised learning
t_prev = 1 #timesteps
t_next = 1
reframed = series_to_supervised(scaled, t_prev, t_next)
print("reframed.shape before drop =", reframed.shape)
#print("\n", reframed.head())
# drop columns we don't want to predict
reframed.drop(reframed.columns[[15,16,17,18,19,20,21,22,23,24,25,26,27,28]], axis=1, inplace=True)
print("reframed.shape after  drop =", reframed.shape)
#print("\n", reframed.head())
print()

#############################################################################

myX = reframed.values #Input data
myN = t_prev

# splitting the data into train/dev and test sets (taking 60% of the data for train/dev)
n_train = 89 * int( myX.shape[0]/89 * 0.60) #each time series has 89 entities

# inputs
train_X = np.array(myX[:n_train, :])
test_X  = np.array(myX[n_train:, :])
# outpts (making sure outputs are arrays of integers)
last_column = values.shape[1] - 1 #last column (goal) in the raw dataset
train_Y = np.array(values[myN:n_train+myN, last_column], dtype=int)
test_Y  = np.array(values[myN+n_train:   , last_column], dtype=int)

# Lets convert Y from its current shape (:,1) into a "one-hot representation" (:,3),
# to make it suitable for the softmax classifier.
def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y
train_Y = convert_to_one_hot(train_Y, C = 3)
test_Y  = convert_to_one_hot(test_Y,  C = 3)
# Now each row in Y will be a one-hot vector representing each class (0, 1, or 2)

# reshape input data to be 3D for Keras [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X  = test_X.reshape( (test_X.shape[0],  1,  test_X.shape[1]))
print("train_X.shape =", train_X.shape)
print("test_X.shape  =", test_X.shape)
print("train_Y.shape =", train_Y.shape)
print("test_Y.shape  =", test_Y.shape)
print()

#############################################################################
#building the model in Keras

from keras import optimizers
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation

# designing the network
input_shape = (train_X.shape[1], train_X.shape[2])

model = Sequential()
model.add(LSTM(32, input_shape=input_shape))
model.add(Dense(3, activation='softmax'))
#model.add(Activation('softmax'))

opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)
### optimizer='adam'
### loss='categorical_accuracy', 'msle'
### metrics=['accuracy'], ['msle', 'mae']
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# fitting the network
#Trained = model.fit(train_X, train_Y, epochs=5, validation_data=(test_X, test_Y), verbose=2, shuffle=False)
Trained = model.fit(train_X, train_Y, epochs=4, validation_split = 0.2, verbose=2, shuffle=False)

# plotting history
pyplot.xlabel('Epoch')
pyplot.ylabel('Loss')
pyplot.plot(Trained.history['loss'],     label='train')
pyplot.plot(Trained.history['val_loss'], label='eval')
pyplot.legend()
pyplot.show()

#summary of the model
model.summary()

#############################################################################
#evaluating the model

from sklearn.metrics import confusion_matrix, classification_report

Test = model.evaluate( x=test_X, y=test_Y, verbose=1 )
print()
print("Test Loss: ", Test[0])
print("Accuracy:  ", Test[1])
print()

Predict = model.predict(x=test_X)
#print("Predict.shape =", Predict.shape)

Y_pred = np.argmax(Predict, axis=1)
Y_true = np.argmax(test_Y,  axis=1)

#for i in range(150,160):
#    print(Y_pred[i], Y_true[i])

print("Confusion Matrix:")
print(confusion_matrix(Y_true, Y_pred))

print("\n Classification Report:")
target_classes = ['No event 0 (-)', 'Event 1 (+)', 'Event 2 (+)']
print(classification_report(Y_true, Y_pred, target_names=target_classes))
