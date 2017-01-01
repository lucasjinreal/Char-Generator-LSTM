import numpy as np
import mxnet as mx
import logging


data_x = np.random.random_sample((122, 5))
print(data_x)
data_y = np.random.random_sample((122, 1))
print(data_y)

x_sym = mx.symbol.Variable('data')
y_sym = mx.symbol.Variable('softmax_label')
print(x_sym)

fc1 = mx.symbol.FullyConnected(data=x_sym, num_hidden=1, name='pre')
loss = mx.symbol.LinearRegressionOutput(data=fc1, label=y_sym, name='loss')

model = mx.model.FeedForward(
    ctx=mx.cpu(),
    symbol=loss,
    num_epoch=100,
    optimizer='adam'
)
logging.basicConfig(level=logging.INFO)

train_iter = mx.io.NDArrayIter(
    data=data_x,
    label=data_y,
    batch_size=10,
    shuffle=True,
)
model.fit(
    X=train_iter,
    eval_metric='mse',
    eval_data=train_iter
)

data_test = np.random.random_sample((10, 5))
predict = model.predict(X=data_test)
print(predict)