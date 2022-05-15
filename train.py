from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import pickle

def print_metrics(mode, clf, features, labels, prob_thresh=0.5):
    probs = clf.predict_proba(features)
    pred = (probs[:, 1] >= prob_thresh).astype(int)
    f1 = f1_score(labels, pred)
    acc = accuracy_score(labels, pred)
    precision = precision_score(labels, pred)
    recall = recall_score(labels, pred)
    print(f'{mode} precision score {precision: .4f}')
    print(f'{mode} recall {recall: .4f}')
    print(f'{mode} accuracy {acc: .4f}')
    print(f'{mode} F1 score {f1: .4f}')
    print()
    return precision, recall, acc, f1


def train_classifier(clf_type, data):
    train_features, train_labels = data['train']
    test_features, test_labels = data['test']
    if clf_type == 'RF':
        for prob_thresh in [0.45, 0.5]:
            for n_estimators in [200, 250, 300, 350]: ## best was 300/350, 8/7, 0.45 and then normal is 200, 7, 0.5
                for max_depth in [8, 9]:
                    print(clf_type, prob_thresh, n_estimators, max_depth)
                    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight='balanced')
                    clf.fit(train_features, train_labels)
                    precision, recall, test_f1, acc = print_metrics('test', clf, test_features, test_labels, prob_thresh=prob_thresh)
    else:
        # class = 'balanced'
        for prob_thresh in [0.45, 0.5, 0.55]:
            for hidden_dim in [50, 100, 150, 200, 250, 300]:
                print(clf_type, prob_thresh, hidden_dim)
                clf = MLPClassifier(hidden_layer_sizes=hidden_dim)
                clf.fit(train_features, train_labels)
                precision, recall, acc, f1 = print_metrics('test', clf, test_features, test_labels, prob_thresh=prob_thresh)


def train_model(data):
    ##################################################################33 cross validation?
    # Random Forest:
    # train_classifier('RF', data)
    # Logistic Reg:
    # train_classifier('MLP', data)
    train_features, train_labels = data['train']
    test_features, test_labels = data['test']
    clf = RandomForestClassifier(n_estimators=350, max_depth=9, class_weight='balanced')
    clf.fit(train_features, train_labels)
    precision, recall, test_f1, acc = print_metrics('test', clf, test_features, test_labels, prob_thresh=0.45)
    pickle.dump(clf, open("clf.pkl", 'wb'))



