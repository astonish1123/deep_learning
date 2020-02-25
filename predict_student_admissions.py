import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_points(data):
    X = np.array(data[["gre", "gpa"]])
    y = np.array(data['admit'])
    admitted = X[np.argwhere(y==1)]
    rejected = X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s=25, color='red', edgecolor='k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s=25, color='cyan', edgecolor='k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def error_formula(y, output):
    return -y*np.log(output) - (1-y)*np.log(1-output)

def error_term_formula(x, y, output):
    return (y - output)*sigmoid_prime(x)

if __name__ == '__main__':
    data = pd.read_csv('student_data.csv')
    plot_points(data)

    data_rank1 = data[data["rank"]==1]
    data_rank2 = data[data["rank"]==2]
    data_rank3 = data[data["rank"]==3]
    data_rank4 = data[data["rank"]==4]

    # plot_points(data_rank1)
    # plt.title("Rank 1")
    # plt.show()
    # plot_points(data_rank2)
    # plt.title("Rank 2")
    # plt.show()
    # plot_points(data_rank3)
    # plt.title("Rank 3")
    # plt.show()
    # plot_points(data_rank4)
    # plt.title("Rank 4")
    # plt.show()

    one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)
    one_hot_data = one_hot_data.drop('rank', axis=1)

    processed_data = one_hot_data[:]
    processed_data['gre'] = processed_data['gre']/800
    processed_data['gpa'] = processed_data['gpa']/4
    print(processed_data[:10])

    sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
    train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)
    # print("Number of training samples is ", len(train_data))
    # print('Number of testing sample is ', len(test_data))
    # print(train_data[:10])
    # print(test_data[:10])

    features = train_data.drop('admit', axis=1)
    targets = train_data['admit']
    features_test = test_data.drop('admit', axis=1)
    targets_test = test_data['admit']
    # print(features_test[:10])
    # print(targets_test[:10])

