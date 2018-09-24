# Deep_Learning_LSTM
Building a classifier model for the Multivariate Time Series data using LSTM in Keras 

This is a three-class classification problem for the given multivariate time series data.
The "goal" column has three values 0 (no event), 1 (event of type 1), and 2 (event of type 2).
Although we can treat events 1 and 2 as one "positive" class and 0 as the "negative" class, we will build a 3-class classifier and create a confusion matrix.

We use LSTM, which is a variation of RNN, for building our network.
