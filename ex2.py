import numpy as np
import sys

# load the input of examples and test x set
train_x, train_y, test_x, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
train_x_arr = np.loadtxt(train_x, delimiter=",")
train_y_arr = np.loadtxt(train_y, delimiter=",").astype(int)
test_x_arr = np.loadtxt(test_x, delimiter=",")
f_out = open(out, 'w')

# find distance between vector a and b
def distance(a, b):
    return np.linalg.norm(a - b)

# normalize the x set with given avg and sd according z-score
def normalize(x_arr, avg_u, sd):
    # normalize according features
    norm_x_arr = x_arr.copy()
    i=0
    for col in norm_x_arr.T:
        col = (col - avg_u[i]) / sd[i]
        norm_x_arr.T[i] = col
        i=i+1
    return norm_x_arr

# shuffle the given examples while maintaining the correct classifications
def shuffle_set(x_arr, y_arr):
    # shuffle the given examples
    shuffled_examples = list(zip(x_arr, y_arr))
    np.random.shuffle(shuffled_examples)
    x_arr = np.array(list(zip(*shuffled_examples))[0])
    y_arr_shuff = np.array(list(zip(*shuffled_examples))[1])
    return x_arr, y_arr_shuff

# computing the hinge loss
def hinge_loss(xi, yi, vector_w, bias):
    # compute the max y which is different from the real y
    wxi_arr = []
    for i in range(3):
        wxi_arr.append(np.dot(vector_w[i], xi)+bias[i])
    wxi_arr[yi]=np.min(wxi_arr)-1
    y_hat = np.argmax(wxi_arr)
    # the hinge loss
    hinge_loss_res = max(0, (1-(np.dot(vector_w[yi], xi)+bias[yi])+(np.dot(vector_w[y_hat], xi)+bias[y_hat])))
    return hinge_loss_res, y_hat

# predict the y of each x with given x set and w
def predict(valid_set, vector_w, bias):
    res_yhats = []
    # predict according the max y_hat
    for xi in valid_set:
        test_vec_y_hats = []
        for i in range(3):
            test_vec_y_hats.append(np.dot(vector_w[i], xi) + bias[i])
        test_y_hat = np.argmax(test_vec_y_hats)
        res_yhats.append(test_y_hat)
    return res_yhats

def knn(x_set, p, k):
    # compute the distance of p and each x in our example set
    dist = []
    for group in range(3):
        for x in x_set[group]:
            euclid_dist = distance(x, p)
            dist.append((euclid_dist, group))

    # sort the distances from the nearest to the farthest
    dist = sorted(dist)[:k]

    # count the frequency of each class among the k nearest neighbors
    class0, class1, class2 = 0, 0, 0
    for d in dist:
        if d[1] == 0:
            class0 += 1
        elif d[1] == 1:
            class1 += 1
        elif d[1] == 2:
            class2 += 1

    # return the class which has the highest frequency
    if class0 > class1:
        if class0 > class2:
            return 0
        return 2
    if class1 > class2:
        return 1
    return 2

def perceptron(zipped_examples, test_x_norm, valid_set_x,valid_set_y, vector_w, bias, eta, epochs):
    # initialization for the best w
    max_success_rate_w = 0
    max_vector_w = np.zeros((3, 5))
    max_bias = np.zeros(3)
    # find the final w vector
    for epoch in range(epochs):
        temp_x_arr, temp_y_arr = shuffle_set(np.array(list(zip(*zipped_examples))[0]), np.array(list(zip(*zipped_examples))[1]))
        zipped_examples = list(zip(temp_x_arr,temp_y_arr))
        for xi, yi in zipped_examples:
            vec_y_hats = []
            for i in range(3):
                vec_y_hats.append(np.dot(vector_w[i], xi)+bias[i])
            y_hat=np.argmax(vec_y_hats)
            if(y_hat!=yi):
                vector_w[yi]=vector_w[yi]+eta*xi
                vector_w[y_hat]=vector_w[y_hat]-eta*xi
                bias[yi]=bias[yi]+eta
                bias[y_hat]=bias[y_hat]-eta

        # send the updated w to validation with the validation set and store this w if it has highest success rate
        valid_yhats=predict(valid_set_x, vector_w, bias)
        success_rate_w = (len([i for i, j in zip(valid_set_y, valid_yhats) if i == j])/len(valid_set_x))*100
        if (success_rate_w>max_success_rate_w):
            max_vector_w = vector_w
            max_bias = bias
            max_success_rate_w = success_rate_w

    # return the predict for each x in test set
    perceptron_yhat = predict(test_x_norm, max_vector_w, max_bias)
    return perceptron_yhat

def svm(zipped_examples, test_x_norm, valid_set_x, valid_set_y, vector_w, bias, eta, lamda, epochs):
    # initialization for the best w
    max_success_rate_w = 0
    max_vector_w = np.zeros((3, 5))
    max_bias = np.zeros(3)
    # find the final w vector
    for epoch in range(epochs):
        temp_x_arr, temp_y_arr = shuffle_set(np.array(list(zip(*zipped_examples))[0]), np.array(list(zip(*zipped_examples))[1]))
        zipped_examples = list(zip(temp_x_arr,temp_y_arr))
        for xi, yi in zipped_examples:
            h_loss, y_hat = hinge_loss(xi, yi, vector_w, bias)
            if(h_loss>0):
                vector_w[yi]=(1-eta*lamda)*vector_w[yi]+eta*xi
                vector_w[y_hat]=(1-eta*lamda)*vector_w[y_hat]-eta*xi
                bias[yi]=(1-eta*lamda)*bias[yi]+eta
                bias[y_hat]=(1-eta*lamda)*bias[y_hat]-eta
                for i in range(3):
                    if((i!=yi)&(i!=y_hat)):
                        vector_w[i]=(1-eta*lamda)*vector_w[i]
                        bias[i]=(1-eta*lamda)*bias[i]
            else:
                for i in range(3):
                        vector_w[i] = (1 - eta * lamda) * vector_w[i]
                        bias[i] = (1 - eta * lamda) * bias[i]

        # send the updated w to validation with the validation set and store this w if it has highest success rate
        valid_yhats=predict(valid_set_x, vector_w, bias)
        success_rate_w = (len([i for i, j in zip(valid_set_y, valid_yhats) if i == j])/len(valid_set_x))*100
        if (success_rate_w>max_success_rate_w):
            max_vector_w = vector_w
            max_bias = bias
            max_success_rate_w = success_rate_w

    # return the predict for each x in test set
    svm_yhat = predict(test_x_norm, max_vector_w, max_bias)
    return svm_yhat

def pa(zipped_examples, test_x_norm, valid_set_x, valid_set_y, vector_w, bias, epochs):
    # initialization for the best w
    max_success_rate_w = 0
    max_vector_w = np.zeros((3, 5))
    max_bias = np.zeros(3)
    # find the final w vector
    for epoch in range(epochs):
        temp_x_arr, temp_y_arr = shuffle_set(np.array(list(zip(*zipped_examples))[0]), np.array(list(zip(*zipped_examples))[1]))
        zipped_examples = list(zip(temp_x_arr,temp_y_arr))
        for xi, yi in zipped_examples:
            xi_norm_2 = 2 * np.power(np.linalg.norm(xi), 2)
            h_loss, y_hat = hinge_loss(xi, yi, vector_w, bias)
            tau = h_loss / xi_norm_2
            vector_w[yi]=vector_w[yi]+tau*xi
            vector_w[y_hat]=vector_w[y_hat]-tau*xi
            bias[yi]=bias[yi]+tau
            bias[y_hat]=bias[y_hat]-tau

        # send the updated w to validation with the validation set and store this w if it has highest success rate
        valid_yhats=predict(valid_set_x, vector_w, bias)
        success_rate_w = (len([i for i, j in zip(valid_set_y, valid_yhats) if i == j])/len(valid_set_x))*100
        if (success_rate_w>max_success_rate_w):
            max_vector_w = vector_w
            max_bias = bias
            max_success_rate_w = success_rate_w

    # return the predict for each x in test set
    pa_yhat=predict(test_x_norm, max_vector_w, max_bias)
    return pa_yhat

def main():
    # normalization
    avg_u = np.average(train_x_arr, axis=0)
    sd = np.sqrt(np.var(train_x_arr, axis=0))
    train_x_norm = normalize(train_x_arr, avg_u, sd)
    test_x_norm = normalize(test_x_arr, avg_u, sd)

    # shuffle the given examples
    train_x_norm, train_y_arr_shuff = shuffle_set(train_x_norm, train_y_arr)

    # separate to train and validation set
    len_train_set = int(0.9 * len(train_x_norm))
    train_set_x = train_x_norm[:len_train_set]
    validation_set_x = train_x_norm[len_train_set:len(train_x_norm)]
    train_set_y = train_y_arr_shuff[:len_train_set]
    validation_set_y = train_y_arr_shuff[len_train_set:len(train_x_norm)]

    # KNN:
    # order into matrix the x train set into their classes
    classes = [[] for i in range(3)]
    i = 0
    for species in train_y_arr_shuff:
        classes[species].append(train_x_norm[i])
        i = i + 1

    # test phase KNN Algorithm
    knn_yhat, j, k = [[] for i in range(len(test_x_norm))], 0, 5
    for p in test_x_norm:
        knn_yhat[j] = knn(classes, p, k)
        j = j + 1

    # Perceptron
    epochs = 100
    eta = 1
    zipped_examples = list(zip(train_set_x, train_set_y))
    vector_w = np.zeros((3,5))
    bias = np.zeros(3)
    perceptron_yhat = perceptron(zipped_examples, test_x_norm, validation_set_x, validation_set_y, vector_w, bias, eta, epochs)

    # SVM
    epochs = 100
    eta = 1
    lamda = 0.0001
    zipped_examples=list(zip(train_set_x, train_set_y))
    vector_w = np.zeros((3,5))
    bias = np.zeros(3)
    svm_yhat = svm(zipped_examples, test_x_norm, validation_set_x, validation_set_y, vector_w, bias, eta, lamda, epochs)

    # PA
    epochs = 100
    zipped_examples=list(zip(train_set_x, train_set_y))
    vector_w = np.zeros((3, 5))
    bias = np.zeros(3)
    pa_yhat = pa(zipped_examples, test_x_norm, validation_set_x, validation_set_y, vector_w, bias, epochs)

    # write to output file
    for i in range(len(test_x_norm)):
        f_out.write(f"knn: {knn_yhat[i]}, perceptron: {perceptron_yhat[i]}, svm: {svm_yhat[i]}, pa: {pa_yhat[i]}\n")


if __name__ == '__main__':
    main()



