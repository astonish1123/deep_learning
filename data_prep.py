import numpy as np
import pandas as pd

if __name__ == '__main__':
    admissions = pd.read_csv('student_data.csv')
    data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank_')], axis=1)
    data = data.drop('rank', axis=1)

    for field in ['gre', 'gpa']:
        mean, std = data[field].mean(), data[field].std()
        data.loc[:, field] = (data[field]-mean)/std

    np.random.seed(42)
    sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
    data, test_data = data.ix[sample], data.drop(sample)

    features, targets = data.drop('admit', axis=1), data['admit']
    features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']