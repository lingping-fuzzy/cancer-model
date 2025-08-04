import torch

import numpy as np
import scipy as sci
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from featureModel.util.funcs import get_sample, center_data, get_SVD, _fit_transform, classify_, classify_probs
from featureModel.utils import lesion_type_dict


def test_effent():
    # # Load the saved features and labels
    source = 'features/retrain_'
    tic_ = ''
    prefix = 'resnet' #effnet
    train_features = torch.load(source+ prefix+'_feature_train'+tic_+'.pt')
    train_labels = torch.load(source+prefix+'_label_train'+tic_+'.pt')
    pred_labels = torch.load(source+prefix+'_preds_train'+tic_+'.pt')

    train_features_val = torch.load(source+prefix+'_feature_val'+tic_+'.pt')
    train_labels_val = torch.load(source+prefix+'_label_val'+tic_+'.pt')
    pred_labels_val = torch.load(source+prefix+'_preds_val'+tic_+'.pt')

    train_features_test = torch.load(source+prefix+'_feature_test'+tic_+'.pt')
    train_labels_test = torch.load(source+prefix+'_label_test'+tic_+'.pt')
    pred_labels_test = torch.load(source+prefix+'_preds_test'+tic_+'.pt')
    # probs_val = torch.load('/kaggle/input/valcut/'+'effnet_probs_val'+tic_+'.pt')
    probs_val = torch.load(source+prefix+'_probs_val'+tic_+'.pt')
    probs_test = torch.load(source+prefix+'_probs_test'+tic_+'.pt')
    # 32*40960,
    # 32*28672
    # 32*20480
    # 32*64

    # train_features = train_features[:, :40960]
    # train_features_val = train_features_val[:, :40960]
    # train_features_test = train_features_test[:, :40960]

    # train_features = train_features[:, 40960:]
    # train_features_val = train_features_val[:, 40960:]
    # train_features_test = train_features_test[:, 40960:]
    y2 = 40960 + 28672
    y3 = 40960 + 28672 + 20480


    # train_features = train_features[:, :y2]
    # train_features_val = train_features_val[:, :y2]
    # train_features_test = train_features_test[:, :y2]

    # remove one layer
    # train_features_val =  torch.cat((train_features_val[:, 0:40960], train_features_val[:, y2:]), dim=1)
    # train_features_test =  torch.cat((train_features_test[:, 0:40960], train_features_test[:, y2:]), dim=1)
    # train_features =  torch.cat((train_features[:, 0:40960], train_features[:, y2:]), dim=1)

    # train_features_val =  torch.cat((train_features_val[:, 0:y2], train_features_val[:, y3:]), dim=1)
    # train_features_test =  torch.cat((train_features_test[:, 0:y2], train_features_test[:, y3:]), dim=1)
    # train_features =  torch.cat((train_features[:, 0:y2], train_features[:, y3:]), dim=1)


    # just use one layer
    # train_features = train_features[:, 40960:y2]
    # train_features_val = train_features_val[:, 40960:y2]
    # train_features_test = train_features_test[:, 40960:y2]
    # train_features = train_features[:, y2:y3]
    # train_features_val = train_features_val[:, y2:y3]
    # train_features_test = train_features_test[:, y2:y3]




    train_features = train_features[:, y3:]
    train_features_val = train_features_val[:, y3:]
    train_features_test = train_features_test[:, y3:]




    # Get the mask where both train_labels and predict_labels are 4
    mask_4 = (train_labels == 4) & (pred_labels ==4)
    # Get the indices where the mask is True
    indices_4 = torch.where(mask_4)[0]

    n_clusters = 500
    # Randomly select 500 indices
    if len(indices_4) > n_clusters:
        # random select
        feature_reduced = train_features[indices_4]
        selected_indices = get_sample(method='clusterpca', feature_reduced=feature_reduced, n_clusters=n_clusters, indices_4=indices_4)
    else:
        selected_indices = indices_4

    # Create a new mask to keep only the selected 500 indices and all other data
    final_mask = torch.ones(len(train_labels), dtype=torch.bool)
    final_mask[indices_4] = False
    final_mask[selected_indices] = True

    # Apply the mask to train_features, train_labels, and predict_labels
    filtered_train_features = train_features[final_mask]
    filtered_train_labels = train_labels[final_mask]
    filtered_predict_labels = pred_labels[final_mask]
    print('finished sampling ---')
    train_feat_centered, val_feat_centered, scaler = center_data(filtered_train_features, train_features_val)
    _train_grouped_proj, proj_M = get_SVD(train_feat_centered, filtered_train_labels)

    _unique_labels = np.unique(filtered_train_labels.numpy())

    test_features_centered = scaler.transform(train_features_test.numpy())
    _proj_val = _fit_transform(proj_M, val_feat_centered)
    _proj_test = _fit_transform(proj_M, test_features_centered)

    _proj_preds_ = classify_(_train_grouped_proj, _proj_val)
    _acc = accuracy_score(train_labels_val, _proj_preds_)

    _test_probs_= classify_probs(_train_grouped_proj, _proj_test, _unique_labels)
    acc =  accuracy_score(train_labels_val, pred_labels_val)

    probs_ = np.array(_test_probs_) * _acc + np.array(probs_test * acc)
    predicted_label_index = np.argmax(probs_, axis=1)
    predicted_label = _unique_labels[predicted_label_index]
    acc_c = accuracy_score(train_labels_test, predicted_label)
    print('val: acc from model: ', acc, '  _acc from features: ', _acc)
    print('combine: ', acc_c)
    print('model test ', accuracy_score(train_labels_test, pred_labels_test))

def test_resnet():
    print('resnet')
    # # Load the saved features and labels
    source = 'features/' #features/retrain_
    tic_ = ''
    prefix = 'resnet' #effnet
    train_features = torch.load(source+ prefix+'_feature_train'+tic_+'.pt')
    train_labels = torch.load(source+prefix+'_label_train'+tic_+'.pt')
    pred_labels = torch.load(source+prefix+'_preds_train'+tic_+'.pt')

    train_features_val = torch.load(source+prefix+'_feature_val'+tic_+'.pt')
    train_labels_val = torch.load(source+prefix+'_label_val'+tic_+'.pt')
    pred_labels_val = torch.load(source+prefix+'_preds_val'+tic_+'.pt')

    train_features_test = torch.load(source+prefix+'_feature_test'+tic_+'.pt')
    train_labels_test = torch.load(source+prefix+'_label_test'+tic_+'.pt')
    pred_labels_test = torch.load(source+prefix+'_preds_test'+tic_+'.pt')
    # probs_val = torch.load('/kaggle/input/valcut/'+'effnet_probs_val'+tic_+'.pt')
    probs_val = torch.load(source+prefix+'_probs_val'+tic_+'.pt')
    probs_test = torch.load(source+prefix+'_probs_test'+tic_+'.pt')

    # 64-64; 32-32; 16-16; linear-64 = 5376+64 = 5440
    y2 = 4096
    y3 = 4096 + 1024
    y4 = y3 + 256

    # train_features = train_features[:, y2:y4]
    # train_features_val = train_features_val[:, y2:y4] -- good results
    # train_features_test = train_features_test[:, y2:y4]

    train_features = train_features[:, :y3]
    train_features_val = train_features_val[:, :y3]
    train_features_test = train_features_test[:, :y3]




    # Get the mask where both train_labels and predict_labels are 4
    mask_4 = (train_labels == 4) & (pred_labels ==4)
    # Get the indices where the mask is True
    indices_4 = torch.where(mask_4)[0]

    n_clusters = 500
    # Randomly select 500 indices
    if len(indices_4) > n_clusters:
        # random select
        feature_reduced = train_features[indices_4]
        selected_indices = get_sample(method='clusterpca', feature_reduced=feature_reduced, n_clusters=n_clusters, indices_4=indices_4)
    else:
        selected_indices = indices_4

    # Create a new mask to keep only the selected 500 indices and all other data
    final_mask = torch.ones(len(train_labels), dtype=torch.bool)
    final_mask[indices_4] = False
    final_mask[selected_indices] = True

    # Apply the mask to train_features, train_labels, and predict_labels
    filtered_train_features = train_features[final_mask]
    filtered_train_labels = train_labels[final_mask]
    filtered_predict_labels = pred_labels[final_mask]
    print('finished sampling ---')
    train_feat_centered, val_feat_centered, scaler = center_data(filtered_train_features, train_features_val)
    _train_grouped_proj, proj_M = get_SVD(train_feat_centered, filtered_train_labels)

    _unique_labels = np.unique(filtered_train_labels.numpy())

    test_features_centered = scaler.transform(train_features_test.numpy())
    _proj_val = _fit_transform(proj_M, val_feat_centered)
    _proj_test = _fit_transform(proj_M, test_features_centered)

    _proj_preds_ = classify_(_train_grouped_proj, _proj_val)
    _acc = accuracy_score(train_labels_val, _proj_preds_)
    print(classification_report(train_labels_val, _proj_preds_, target_names=lesion_type_dict))

    _test_probs_= classify_probs(_train_grouped_proj, _proj_test, _unique_labels)
    acc =  accuracy_score(train_labels_val, pred_labels_val)
    print(classification_report(train_labels_val, pred_labels_val, target_names=lesion_type_dict))

    probs_ = np.array(_test_probs_) * _acc + np.array(probs_test * acc)
    predicted_label_index = np.argmax(probs_, axis=1)
    predicted_label = _unique_labels[predicted_label_index]
    acc_c = accuracy_score(train_labels_test, predicted_label)

    print('val: acc from model: ', acc, '  _acc from features: ', _acc)
    print('combine: ', acc_c)
    print('model test ', accuracy_score(train_labels_test, pred_labels_test))

    print(classification_report(train_labels_test, np.argmax(probs_test, axis=1), target_names=lesion_type_dict))
    print(classification_report(train_labels_test, pred_labels_test, target_names=lesion_type_dict))

if __name__ == "__main__":
    test_resnet()


'''resnet train_features
finished sampling ---
val: acc from model:  0.8526785714285714   _acc from features:  0.7660714285714286
combine:  0.8697591436217663
model test  0.8706512042818911

Process finished with exit code 0

resnet - train_features[:, y2:y3]
finished sampling ---
val: acc from model:  0.8526785714285714   _acc from features:  0.5392857142857143
combine:  0.872435325602141
model test  0.8706512042818911

resnet
finished sampling --- train_features[:, :y3]
val: acc from model:  0.8526785714285714   _acc from features:  0.48482142857142857
combine:  0.8706512042818911
model test  0.8706512042818911

Process finished with exitresnet
finished sampling ---train_features[:, y3:]
val: acc from model:  0.8526785714285714   _acc from features:  0.8116071428571429
combine:  0.8715432649420161
model test  0.8706512042818911

Process finished with exit code 0 code 0


----source 
finished sampling ---train_features[:, y2:y3]
val: acc from model:  0.8455357142857143   _acc from features:  0.5517857142857143
combine:  0.8572702943800179
model test  0.8581623550401427

finished sampling ---train_features[:, :y2]
val: acc from model:  0.8455357142857143   _acc from features:  0.5758928571428571
combine:  0.8572702943800179
model test  0.8581623550401427
'''