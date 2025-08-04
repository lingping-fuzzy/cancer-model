
import torch
import numpy as np


from featureModel.util.funcs import get_sample, center_data, get_SVD, _fit_transform, classify_, classify_probs
from featureModel.utils import lesion_type_dict

from sklearn.metrics import classification_report
import pandas as pd

def classification_report_to_latex(y_true, y_pred, target_names=None):
    # Get classification report as a dictionary
    report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)

    # Convert the dictionary into a DataFrame for easier manipulation
    report_df = pd.DataFrame(report_dict).transpose()

    # Initialize the LaTeX table code
    latex_str = r'\begin{table}[h!]\n\centering\n\begin{tabular}{lcccc}\n'
    latex_str += r'\hline\n'
    latex_str += r'Class & Precision & Recall & F1-Score & Support \\\\ \n'
    latex_str += r'\hline\n'

    # Add rows for each class and average metrics
    for idx, row in report_df.iterrows():
        if isinstance(idx, str) and idx in ["accuracy"]:
            continue  # Skip accuracy as it's a single value, we'll print it separately
        latex_str += f'{idx} & {row["precision"]:.2f} & {row["recall"]:.2f} & {row["f1-score"]:.2f} & {int(row["support"])} \\\\ \n'

    # Add accuracy separately
    latex_str += r'\hline\n'
    latex_str += f'Accuracy & & & {report_df.loc["accuracy"]["precision"]:.2f} & {int(report_df.loc["accuracy"]["support"])} \\\\ \n'
    latex_str += r'\hline\n'

    # End the LaTeX table
    latex_str += r'\end{tabular}\n\caption{Classification Report}\n\end{table}'

    return latex_str


def feature_effent(x=0,y=0, mod_re = True):
    # # Load the saved features and labels
    source = 'features/retrain_'  #
    tic_ = ''
    prefix = 'effnet'

    train_features = torch.load(source+ prefix+'_feature_train'+tic_+'.pt')
    train_labels = torch.load(source+prefix+'_label_train'+tic_+'.pt')
    pred_labels = torch.load(source+prefix+'_preds_train'+tic_+'.pt')

    train_features_val = torch.load(source+prefix+'_feature_val'+tic_+'.pt')


    train_features_test = torch.load(source+prefix+'_feature_test'+tic_+'.pt')
    train_labels_test = torch.load(source+prefix+'_label_test'+tic_+'.pt')

    train_features = train_features[:, x:y]
    train_features_val = train_features_val[:, x:y]
    train_features_test = train_features_test[:, x:y]


    filtered_train_labels, filtered_predict_labels, filtered_train_features = mask_data(train_labels, pred_labels,
                                                                                        n_clusters=300,
                                                                                        data_features=train_features)

    # Apply the mask to train_features, train_labels, and predict_labels

    train_feat_centered, val_feat_centered, scaler = center_data(filtered_train_features, train_features_val)
    _train_grouped_proj, proj_M = get_SVD(train_feat_centered, filtered_train_labels)

    test_features_centered = scaler.transform(train_features_test.numpy())
    _proj_test = _fit_transform(proj_M, test_features_centered)

    np.save('effnet_library.npy',  _train_grouped_proj)
    np.save('effnet_library_test.npy', _proj_test)
    np.save('effnet_library_label.npy', filtered_train_labels)
    np.save('effnet_library_test_label.npy', train_labels_test)

def getmask_size(train_labels, pred_labels):
    # Get the mask where both train_labels and predict_labels are 4
    _min = 5000
    for i in range(7):
        mask_ = (train_labels == i) & (pred_labels == i)
        if _min > torch.sum(mask_).item():
            _min = torch.sum(mask_).item()
    # Get the indices where the mask is True
    return _min

def mask_data(train_labels, pred_labels, n_clusters, data_features):
    # Get the mask where both train_labels and predict_labels are 4
    mask_4 = (train_labels == 4) & (pred_labels == 4)
    # Get the indices where the mask is True
    indices_4 = torch.where(mask_4)[0]

    # Randomly select 500 indices
    if len(indices_4) > n_clusters:
        # random select
        feature_reduced = data_features[indices_4]
        selected_indices = get_sample(method='clusterpca', feature_reduced=feature_reduced, n_clusters=n_clusters,
                                      indices_4=indices_4)
    else:
        selected_indices = indices_4

    # Create a new mask to keep only the selected 500 indices and all other data
    final_mask = torch.ones(len(train_labels), dtype=torch.bool)
    final_mask[indices_4] = False
    final_mask[selected_indices] = True

    for i in [0, 1, 2, 3, 5, 6]:
        mask_4 = (train_labels == i) & (pred_labels == i)
        indices_4 = torch.where(mask_4)[0]
        mask = (train_labels == i)
        final_mask[torch.where(mask)[0]] = False
        final_mask[indices_4] = True


    # Apply the mask to train_features, train_labels, and predict_labels
    filtered_train_features = data_features[final_mask]
    filtered_train_labels = train_labels[final_mask]
    filtered_predict_labels = pred_labels[final_mask]
    return  filtered_train_labels, filtered_predict_labels, filtered_train_features


def feature_resnet(x = 0, y=0, mod_re = True):
    print('resnet')
    # # Load the saved features and labels
    tic_ = ''
    prefix = 'resnet'  # effnet
    source = 'features/retrain_'  # features

    train_features = torch.load(source+ prefix+'_feature_train'+tic_+'.pt')
    train_labels = torch.load(source+prefix+'_label_train'+tic_+'.pt')
    pred_labels = torch.load(source+prefix+'_preds_train'+tic_+'.pt')

    train_features_val = torch.load(source+prefix+'_feature_val'+tic_+'.pt')

    train_features_test = torch.load(source+prefix+'_feature_test'+tic_+'.pt')
    train_labels_test = torch.load(source+prefix+'_label_test'+tic_+'.pt')


    train_features = train_features[:, x:y]
    train_features_val = train_features_val[:, x:y]
    train_features_test = train_features_test[:, x:y]


    # from here, we have presentation  train_features,
    #  mask_data is to find instances that are corrected classified and balance the data (reduce too many instances from nv)
    filtered_train_labels, filtered_predict_labels, filtered_train_features = mask_data(train_labels, pred_labels, n_clusters=300, data_features=train_features)

    # print('finished sampling ---')
    train_feat_centered, val_feat_centered, scaler = center_data(filtered_train_features, train_features_val)
    #  representative category-representation-library- > _train_grouped_proj
    _train_grouped_proj, proj_M = get_SVD(train_feat_centered, filtered_train_labels)

    _unique_labels = np.unique(filtered_train_labels.numpy())
    test_features_centered = scaler.transform(train_features_test.numpy())
    _proj_test = _fit_transform(proj_M, test_features_centered)

    _proj_preds_ = classify_(_train_grouped_proj, _proj_test)

    np.save('resnet_library.npy',  _train_grouped_proj)
    np.save('resnet_library_test.npy', _proj_test)
    np.save('resnet_library_label.npy', filtered_train_labels)
    np.save('resnet_library_test_label.npy', train_labels_test)


if __name__ == "__main__":

    # 64-64; 32-32; 16-16; linear-64 = 5376+64 = 5440
    y2 = 4096
    y3 = 4096 + 1024
    y4 = y3 + 256
    print(' This is result for the resnet with retrained model')
    feature_resnet(x=0, y=y2, mod_re=True)



    y2 = 40960
    y3 = 40960 + 28672
    y4 = 40960 + 28672 + 20480
    print(' This is result for the effnet with retrained model')
    feature_effent(x=y2, y=y3,mod_re = True)




