import codecs
import numpy as np
import os


def load_data(data_dir):
    data_tensors = {}
    for fname in ('train', 'test'):
        print('reading', fname)
        data_tensors[fname] = np.loadtxt(os.path.join(data_dir, fname + '.txt'))
    return data_tensors


def get_batch(iterable, batch=1):
    l = len(iterable)
    for index in range(0, l, batch):
        yield iterable[index:min(index + batch, l)]


class DataReader:

    def __init__(self, data_tensor, input_dim, output_dim, batch_size):
        print('get in')
        inputs = data_tensor[:, input_dim]
        outputs = data_tensor[:, output_dim]
        print(inputs.shape)
        print(outputs.shape)
        x_batches = []
        y_batches = []
        for xs in get_batch(inputs, batch_size):
            x_batches.append(xs)
        for ys in get_batch(outputs, batch_size):
            y_batches.append(ys)
        print(len(x_batches), x_batches[0].shape)
        print(len(y_batches), y_batches[0].shape)
        self._x_batches = x_batches
        self._y_batches = y_batches
        self.batch_size = batch_size
        print(len(self._x_batches), self._x_batches[0].shape)

    def iter(self):
        for x, y in zip(self._x_batches, self._y_batches):
            yield x, y


if __name__ == '__main__':
    data = load_data('Data')
    print('print sample of input and output')
    count = 0
    for i, j in DataReader(data['test'], [0, 1], [5], 100).iter():
        count += 1
        if count == 5:
            print(i, j)
