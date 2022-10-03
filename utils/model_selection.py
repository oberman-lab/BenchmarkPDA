import numpy as np
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F

def entropy(input_):
    entropy = -input_ * torch.log(input_ + 1e-7)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def image_classification(loader, model):
    model.train(False)
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            _, outputs = model(inputs)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(entropy(torch.nn.Softmax(dim=1)(all_output))).cpu().data.item()

    hist_tar = torch.nn.Softmax(dim=1)(all_output).sum(dim=0)
    hist_tar = hist_tar / hist_tar.sum()
    return accuracy, hist_tar, mean_ent, all_output, all_label


def get_acc_10crop(loaders: dict, model):
    model.train(False)
    start_test = True
    with torch.no_grad():
        iter_test = [iter(loaders[i]) for i in range(10)]
        for i in range(len(loaders[0])):
            data = [iter_test[j].next() for j in range(10)]
            inputs = [data[j][0].cuda() for j in range(10)]
            labels = data[0][1]
            softmaxes = []
            for j in range(10):
                _, logits_temp = model(inputs[j])
                softmaxes.append(torch.nn.Softmax(dim=1)(logits_temp))
            softmaxes = sum(softmaxes)/len(softmaxes)
            if start_test:
                all_softmaxes = softmaxes.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_softmaxes = torch.cat((all_softmaxes, softmaxes.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    _, predict = torch.max(all_softmaxes, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    return accuracy


def get_data(loader, model):
    model.train(False)
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            features, logits = model(inputs)
            if start_test:
                all_logits = logits.float().cpu()
                all_features = features.float().cpu()
                all_labels = labels.float()
                start_test = False
            else:
                all_features = torch.cat((all_features, features.float().cpu()), 0)
                all_logits = torch.cat((all_logits, logits.float().cpu()), 0)
                all_labels = torch.cat((all_labels, labels.float()), 0)    
    return all_features, all_logits, all_labels


def get_data_limited(loader, model, limit=3000):
    model.train(False)
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            features, logits = model(inputs)
            if start_test:
                all_logits = logits.float().cpu()
                all_features = features.float().cpu()
                all_labels = labels.float()
                start_test = False
            else:
                all_features = torch.cat((all_features, features.float().cpu()), 0)
                all_logits = torch.cat((all_logits, logits.float().cpu()), 0)
                all_labels = torch.cat((all_labels, labels.float()), 0)    
                if len(all_features)>3000:
                    break
    return all_features, all_logits, all_labels


def get_acc(logits, labels):
    _, predict = torch.max(logits, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == labels).item() / float(labels.size()[0])
    return accuracy

def get_mean_ent(logits):
    mean_ent = torch.mean(entropy(torch.nn.Softmax(dim=1)(logits))).cpu().data.item()
    return mean_ent

def get_class_weight(logits):
    class_weight = torch.nn.Softmax(dim=1)(logits).sum(dim=0)
    class_weight = class_weight / class_weight.sum()
    return class_weight.cuda().detach()

# Adapted from https://github.com/VisionLearningGroup/SND
def get_snd(logits):
    softmaxes = torch.nn.Softmax(dim=1)(logits)
    normalized = F.normalize(softmaxes).cpu()
    mat = torch.matmul(normalized, normalized.t()) / 0.05
    mask = torch.eye(mat.size(0), mat.size(0)).bool()
    mat.masked_fill_(mask, -1 / 0.05)
    snd = torch.mean(entropy(torch.nn.Softmax(dim=1)(mat))).item()
    return snd

def get_error(logits, labels):
    error = 1-(logits.argmax(dim=1, keepdim=True) == labels.reshape(-1,1)).float().numpy()
    return error

# Adapted from https://github.com/thuml/TransCal
def get_importance_weights_lr(source_feature, target_feature, validation_feature, random_state):
    """
    :param source_feature: shape [N_tr, d], features from training set
    :param target_feature: shape [N_te, d], features from test set
    :param validation_feature: shape [N_v, d], features from validation set
    :return:
    """
    N_s, d = source_feature.shape
    N_t, _d = target_feature.shape
    source_feature = source_feature.numpy().copy()
    target_feature = target_feature.numpy().copy()
    all_feature = np.concatenate((source_feature, target_feature))
    all_label = np.asarray([1] * N_s + [0] * N_t,dtype=np.int32)
    feature_for_train,feature_for_test, label_for_train,label_for_test = train_test_split(all_feature, all_label, train_size = 0.8, random_state=random_state)
    
    domain_classifier = linear_model.LogisticRegression(solver='liblinear', max_iter=500, random_state=random_state)
    domain_classifier.fit(feature_for_train, label_for_train)
    output = domain_classifier.predict(feature_for_test)
    acc = np.mean((label_for_test == output).astype(np.float32))
    
    domain_out = domain_classifier.predict_proba(validation_feature)
    weight = domain_out[:,:1] / domain_out[:,1:] * N_s * 1.0 / N_t
    return weight


# Adapted from https://github.com/VisionLearningGroup/SND
def get_importance_weights_svm(source_feature, target_feature, validation_feature, random_state):
    """
    :param source_feature: shape [N_tr, d], features from training set
    :param target_feature: shape [N_te, d], features from test set
    :param validation_feature: shape [N_v, d], features from validation set
    :return:
    """
    source_feature = source_feature.numpy().copy()
    target_feature = target_feature.numpy().copy()
    rand_s = np.random.permutation(source_feature.shape[0])
    rand_t = np.random.permutation(target_feature.shape[0])

    source_feature = source_feature[rand_s[:3000]]
    target_feature = target_feature[rand_t[:3000]]
    N_s, d = source_feature.shape
    N_t, _d = target_feature.shape
    all_feature = np.concatenate((source_feature, target_feature))
    all_label = np.asarray([1] * N_s + [0] * N_t, dtype=np.int32)
    feature_for_train, feature_for_test, label_for_train, label_for_test = train_test_split(all_feature, all_label,
                                                                                            train_size=0.8)
    decays = np.logspace(-2, 4, 5)
    val_acc = []
    domain_classifiers = []
    for decay in decays:
        domain_classifier = svm.SVC(C=decay, kernel='linear', verbose=False, probability=True,
                                    max_iter=4000, random_state=random_state)
        domain_classifier.fit(feature_for_train, label_for_train)
        output = domain_classifier.predict(feature_for_test)
        acc = np.mean((label_for_test == output).astype(np.float32))
        val_acc.append(acc)
        domain_classifiers.append(domain_classifier)

    index = val_acc.index(max(val_acc))

    domain_classifier = domain_classifiers[index]

    domain_out = domain_classifier.predict_proba(validation_feature)
    return domain_out[:, :1] / domain_out[:, 1:] * N_s * 1.0 / N_t


# Adapted from https://github.com/thuml/Deep-Embedded-Validation
def get_importance_weights_mlp(source_feature, target_feature, validation_feature, random_state):
    """
    :param source_feature: shape [N_tr, d], features from training set
    :param target_feature: shape [N_te, d], features from test set
    :param validation_feature: shape [N_v, d], features from validation set
    :return:
    """
    N_s, d = source_feature.shape
    N_t, _d = target_feature.shape
    source_feature = source_feature.numpy().copy()
    target_feature = target_feature.numpy().copy()
    all_feature = np.concatenate((source_feature, target_feature))
    all_label = np.asarray([1] * N_s + [0] * N_t,dtype=np.int32)
    feature_for_train,feature_for_test, label_for_train,label_for_test = train_test_split(all_feature, all_label, train_size = 0.8)
    
    decays = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    val_acc = []
    domain_classifiers = []
    
    for decay in decays:
        domain_classifier = MLPClassifier(hidden_layer_sizes=(d, d, 2), activation='relu', 
                                          alpha=decay, max_iter=400, random_state=random_state)
        domain_classifier.fit(feature_for_train, label_for_train)
        output = domain_classifier.predict(feature_for_test)
        acc = np.mean((label_for_test == output).astype(np.float32))
        val_acc.append(acc)
        domain_classifiers.append(domain_classifier)
        
    index = val_acc.index(max(val_acc))

    domain_classifier = domain_classifiers[index]

    domain_out = domain_classifier.predict_proba(validation_feature)
    return domain_out[:,:1] / domain_out[:,1:] * N_s * 1.0 / N_t

# Adapted from https://github.com/thuml/Deep-Embedded-Validation
def get_dev_risk(weight, error):
    """
    :param weight: shape [N, 1], the importance weight for N source samples in the validation set
    :param error: shape [N, 1], the error value for each source sample in the validation set
    (typically 0 for correct classification and 1 for wrong classification)
    """
    N, d = weight.shape
    _N, _d = error.shape
    assert N == _N and d == _d, 'dimension mismatch!'
    weighted_error = weight * error
    cov = np.cov(np.concatenate((weighted_error, weight), axis=1),rowvar=False)[0][1]
    var_w = np.var(weight, ddof=1)
    eta = - cov / var_w
    return np.mean(weighted_error) + eta * np.mean(weight) - eta
