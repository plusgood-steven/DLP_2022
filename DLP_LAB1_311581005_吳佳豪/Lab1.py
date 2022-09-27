# %%
import numpy as np
import matplotlib.pyplot as plt
import math
# %%


def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    inputs = []
    labels = []

    for i in range(11):
        inputs.append([0.1 * i, 0.1 * i])
        labels.append(0)

        if 0.1 * i == 0.5:
            continue

        inputs.append([0.1 * i, 1 - 0.1 * i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def derivative_sigmoid(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def relu(x):
    return np.maximum(x, 0)


def derviative_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title("Ground truth", fontsize=18)
    for i, (x1, y1) in enumerate(x):
        plt.plot(x1, y1, "ro" if y[i] == 0 else "bo")
    plt.subplot(1, 2, 2)
    plt.title("Predict", fontsize=18)
    for i, (x1, y1) in enumerate(x):
        plt.plot(x1, y1, "ro" if pred_y[i] == 0 else "bo")
    plt.show()


def show_linear(x, title=None):
    plt.title(title, fontsize=18)
    plt.plot([i + 1 for i in range(len(x))], x)
    plt.show()


def show_linear_compare(loss, accu):
    plt.subplot(1, 2, 1)
    plt.title("Loss/Epochs", fontsize=18)
    plt.plot([i + 1 for i in range(len(loss))], loss)
    plt.subplot(1, 2, 2)
    plt.title("Accuracy/Epochs", fontsize=18)
    plt.plot([i + 1 for i in range(len(accu))], accu)
    plt.show()


class SGD:
    def __init__(self, lr=0.01, momentum=0.0):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, gards):
        if self.v is None:
            self.v = [np.zeros_like(gard) for gard in gards]
        for layer in range(len(gards)):
            self.v[layer] = -self.lr * gards[layer] + \
                self.momentum * self.v[layer]
            params[layer] += self.v[layer]
        return params

# %%


class NeuralNetwork():
    def __init__(self, layers: list) -> None:
        self.layers = layers
        self.layers_size = len(self.layers)
        self.params = {
            "weight": [],
            "bias": [],
            "a": [],
            "z": [],
        }
        self.gards = {
            "dw": [],
            "db": [],
            "dz": [],
        }
        self.init_layers()

    def init_layers(self):
        for i in range(self.layers_size):
            input_dim = self.layers[i]["input_dim"]
            output_dim = self.layers[i]["output_dim"]
            self.params["weight"].append(
                np.random.normal(size=(output_dim, input_dim)))
            self.params["bias"].append(
                np.random.normal(size=(output_dim, 1)))
            self.params["a"].append(np.random.randn(output_dim, 1) * 0.1)
            self.params["z"].append(np.random.randn(output_dim, 1) * 0.1)

            self.gards["dw"].append(np.ones((output_dim, input_dim)))
            self.gards["db"].append(np.ones((output_dim, 1)))
            self.gards["dz"].append(np.ones((output_dim, 1)))

    def choose_activated_func(self, func_name):
        func = None
        if func_name == "sigmoid":
            func = sigmoid
        elif func_name == "relu":
            func = relu
        else:
            def func(x): return x
        return func

    def choose_derivative_func(self, func_name):
        func = None
        if func_name == "sigmoid":
            func = derivative_sigmoid
        elif func_name == "relu":
            func = derviative_relu
        else:
            def func(x): return 1
        return func

    def MSE(self, y_hat, Y):
        loss = np.square(y_hat - Y).mean()
        return loss

    def derivative_MSE(self, y_hat, Y):
        return 2 * ((y_hat - Y).sum()).mean()

    def classify(self, y_hat):
        pre = np.copy(y_hat)
        for i, num in enumerate(pre[0]):
            pre[0][i] = round(num)
        return np.array(pre, dtype=int).T

    def compute_accurancy(self, y_hat, Y):
        right = 0
        for i, predict in enumerate(y_hat):
            right += 1 if Y[i][0] == predict[0] else 0
        return right / y_hat.shape[0]

    def forward(self, x):
        for i in range(self.layers_size):
            activate_func = self.choose_activated_func(
                self.layers[i]["activate_function"])
            self.params["z"][i] = np.dot(
                self.params["weight"][i], self.params["a"][i - 1] if i != 0 else x) + self.params["bias"][i]
            self.params["a"][i] = activate_func(self.params["z"][i])
        return self.params["a"][self.layers_size - 1]

    def backward(self, x, y_hat, Y):
        for i in range(self.layers_size - 1, -1, -1):
            derivative_activate_func = self.choose_derivative_func(
                self.layers[i]["activate_function"])
            if i == self.layers_size - 1:
                self.gards["dz"][i] = self.derivative_MSE(
                    y_hat, Y) * derivative_activate_func(self.params["z"][i])
            else:
                self.gards["dz"][i] = np.dot(
                    self.params["weight"][i + 1].T, self.gards["dz"][i + 1]) * derivative_activate_func(self.params["z"][i])

            self.gards["dw"][i] = np.dot(self.gards["dz"][i], np.transpose(
                self.params["a"][i - 1] if i != 0 else x)) / Y.shape[1]
            self.gards["db"][i] = np.sum(
                self.gards["dz"][i]) / Y.shape[1]

    def update(self):
        self.params["weight"] = self.weight_optimizer.update(
            self.params["weight"], self.gards["dw"])
        self.params["bias"] = self.bias_optimizer.update(
            self.params["bias"], self.gards["db"])

    def train(self, x_train, y_train, epochs, lr=1.0e-2, momentum=0.0, batch_size=1, early_stop_step=None):
        loss_record = []
        accu_record = []
        max_accu = 0
        early_stop = 0
        self.weight_optimizer = SGD(lr=lr, momentum=momentum)
        self.bias_optimizer = SGD(lr=lr, momentum=momentum)
        print("Params Setting:\nepchos:{epchos} learning rate:{lr:e} momentum:{momentum} batch size:{batch_size} early_stop_step:{early_stop_step}".format(
            epchos=epochs, lr=lr, momentum=momentum, batch_size=batch_size, early_stop_step=early_stop_step))
        print("-" * 10)
        for e in range(1, epochs + 1, 1):
            loss = 0
            final_y = None
            for i in range(math.ceil(x_train.shape[0] / batch_size)):
                upperindex = (i + 1) * batch_size if (i + 1) * \
                    batch_size < x_train.shape[0] else x_train.shape[0]
                x = np.array(x_train[i * batch_size: upperindex]).T
                y = np.array(y_train[i * batch_size: upperindex]).T
                y_hat = self.forward(x)
                loss += self.MSE(y_hat, y) * y.shape[1]
                final_y = np.concatenate((final_y, self.classify(
                    y_hat)), axis=0) if final_y is not None else self.classify(y_hat)
                self.backward(x, y_hat, y)
                self.update()

            accu = self.compute_accurancy(final_y, y_train)
            loss /= y_train.shape[0]
            accu_record.append(accu)
            loss_record.append(loss)

            if(accu > max_accu):
                max_accu = accu
                early_stop = 0
            else:
                early_stop += 1
            if(e % 10 == 0 or accu == 1.0):
                print("epochs: {e} accurancy: {accu:.2%} loss: {loss}".format(
                    e=e, accu=accu, loss=loss))
                if(accu == 1.0 or (early_stop_step and early_stop > early_stop_step)):
                    print("Early Stop !")
                    break

        return final_y, loss_record, accu_record

    def predict(self, x):
        return self.classify(self.forward(x))


# %%
# layer = [{"input_dim": 2, "output_dim": 4, "activate_function": None},
#          {"input_dim": 4, "output_dim": 4, "activate_function": None},
#          {"input_dim": 4, "output_dim": 1, "activate_function": None}]

layer = [{"input_dim": 2, "output_dim": 4, "activate_function": "sigmoid"},
         {"input_dim": 4, "output_dim": 4, "activate_function": "sigmoid"},
         {"input_dim": 4, "output_dim": 1, "activate_function": "sigmoid"}]


print("-"*20, "\nLinear dataset")
x, y = generate_linear(n=100)
NN = NeuralNetwork(layer)
y_hat, loss_record, accu_record = NN.train(
    x, y, epochs=10000, lr=0.05, momentum=0.9, batch_size=1)

show_result(x, y, y_hat)
show_linear_compare(loss_record, accu_record)
# show_linear(loss_record, "Loss/Epochs")
# show_linear(accu_record, "Accuracy/Epochs")


print("-"*20, "\nXOR dataset")
x, y = generate_XOR_easy()
NN = NeuralNetwork(layer)
y_hat, loss_record, accu_record = NN.train(
    x, y, epochs=10000, lr=0.05, momentum=0.9, batch_size=1)

show_result(x, y, y_hat)
show_linear_compare(loss_record, accu_record)

# %%
