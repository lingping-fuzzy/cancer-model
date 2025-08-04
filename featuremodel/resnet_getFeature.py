import torch
from torch.utils.data import DataLoader

from utils import _get_df_data_, dfImageDataset, get_fetures_eff, get_fetures_res, _get_df_cutdata_, get_fetures_prob


def get_loder(cut = True):
    if cut == True:
        train_df, val_df, test_df = _get_df_cutdata_()
    else:
        train_df, val_df, test_df = _get_df_data_()

    train_data = dfImageDataset(train_df)
    val_data = dfImageDataset(val_df)
    test_data = dfImageDataset(test_df)

    BATCH_SIZE = 32
    # Data loaders
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE)
    valloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, valloader, testloader

def save_output(loader, model_path, loder_name= 'val', cut=True, model_name='effnet', add=''):
    if model_name == 'effnet':
        features_, labels_, pred_ = get_fetures_eff(loader, model_path=model_path)
    else:
        features_, labels_, pred_ = get_fetures_res(loader, model_path=model_path)

    # if cut == True:
    #     # Save train_features and train_labels to disk
    #     torch.save(features_, (add+model_name+'_feature_'+loder_name+'_cut.pt'))
    #     torch.save(labels_, (add+model_name+'_label_'+loder_name+'_cut.pt'))
    #     torch.save(pred_, (add+model_name+'_preds_'+loder_name+'_cut.pt'))
    # else:
    #     torch.save(features_, (add+model_name+'_feature_'+loder_name+'.pt'))
    #     torch.save(labels_, (add+model_name+'_label_'+loder_name+'.pt'))
    #     torch.save(pred_, (add+model_name+'_preds_'+loder_name+'.pt'))

def save_prob(loader, model_path, loder_name= 'val', cut=True, model_name='effnet', add=''):

    features_, labels_, pred_ = get_fetures_prob(loader, model_path=model_path, modelname=model_name)

    if cut == True:
        # Save train_features and train_labels to disk
        torch.save(features_, (add+model_name+'_probs_'+loder_name+'_cut.pt'))

    else:
        torch.save(features_, (add+model_name+'_probs_'+loder_name+'.pt'))


model_path = ['trained_models/efficient_combine_model_cut_1.pt', 'trained_models/resnet_combine_model_cut_1.pt',
              'trained_models/res_combine_model_tune.pt', 'trained_models/eff_combine_model_tune.pt',
              'trained_models/retrain_eff_combine_model_tune1.pt', 'trained_models/retrain_resnet_combine_model_layer2.2.pt']
cut_or = True


# trainloader, val_loader, test_loader = get_loder(cut = cut_or)

# save_output(trainloader, model_path[0], loder_name= 'train', cut=cut_or, model_name='effnet')
# save_output(trainloader, model_path[1], loder_name= 'train', cut=cut_or, model_name='resnet')
#
# save_output(val_loader, model_path[0], loder_name= 'val', cut=cut_or, model_name='effnet')
# save_output(val_loader, model_path[1], loder_name= 'val', cut=cut_or, model_name='resnet')

# save_output(test_loader, model_path[0], loder_name= 'test', cut=cut_or, model_name='effnet')
# save_output(test_loader, model_path[1], loder_name= 'test', cut=cut_or, model_name='resnet')


cut_or = False
trainloader, val_loader, test_loader = get_loder(cut = cut_or)

import time
start_time = time.time()
save_output(trainloader, model_path[2], loder_name= 'train', cut=cut_or, model_name='resnet')
print('time cost ', time.time()-start_time)
# save_output(val_loader, model_path[2], loder_name= 'val', cut=cut_or, model_name='resnet')
# save_output(test_loader, model_path[2], loder_name= 'test', cut=cut_or, model_name='resnet')
#
# save_prob(trainloader, model_path[2], loder_name= 'val', cut=cut_or, model_name='resnet')
# save_prob(test_loader, model_path[2], loder_name= 'test', cut=cut_or, model_name='resnet')
#
# save_prob(val_loader, model_path[5], loder_name= 'val', cut=cut_or, model_name='resnet', add='retrain_')
# save_prob(test_loader, model_path[5], loder_name= 'test', cut=cut_or, model_name='resnet', add='retrain_')
#
# save_output(trainloader, model_path[5], loder_name= 'train', cut=cut_or, model_name='resnet', add='retrain_')
# save_output(val_loader, model_path[5], loder_name= 'val', cut=cut_or, model_name='resnet', add='retrain_')
# save_output(test_loader, model_path[5], loder_name= 'test', cut=cut_or, model_name='resnet', add='retrain_')

# trainloader, val_loader, test_loader = get_loder(cut = cut_or)
#
# save_prob(trainloader, model_path[0], loder_name= 'train', cut=cut_or, model_name='effnet')
# save_prob(trainloader, model_path[1], loder_name= 'train', cut=cut_or, model_name='resnet')
# #
# save_prob(val_loader, model_path[0], loder_name= 'val', cut=cut_or, model_name='effnet')
# save_prob(val_loader, model_path[1], loder_name= 'val', cut=cut_or, model_name='resnet')
#
# save_prob(test_loader, model_path[0], loder_name= 'test', cut=cut_or, model_name='effnet')
# save_prob(test_loader, model_path[1], loder_name= 'test', cut=cut_or, model_name='resnet')
#
#
# cut_or = False
# trainloader, val_loader, test_loader = get_loder(cut = cut_or)
#
# save_prob(trainloader, model_path[3], loder_name= 'train', cut=cut_or, model_name='effnet')
# save_prob(trainloader, model_path[2], loder_name= 'train', cut=cut_or, model_name='resnet')
#
# save_prob(val_loader, model_path[3], loder_name= 'val', cut=cut_or, model_name='effnet')
# save_prob(val_loader, model_path[2], loder_name= 'val', cut=cut_or, model_name='resnet')
#
# save_prob(test_loader, model_path[3], loder_name= 'test', cut=cut_or, model_name='effnet')
# save_prob(test_loader, model_path[2], loder_name= 'test', cut=cut_or, model_name='resnet')



# features_, labels_, pred_  = get_fetures_eff(train_loader, model_path='trained_models/efficient_combine_model_cut_1.pt')
# # Save train_features and train_labels to disk
# torch.save(features_, 'eff_train_features_cut.pt')
# torch.save(labels_, 'eff_train_labels_cut.pt')
# torch.save(pred_, 'res_predict_labels_cut.pt')
#
# features_, labels_, pred_ = get_fetures_res(train_loader, model_path='trained_models/resnet_combine_model_cut_1.pt')
# # Save train_features and train_labels to disk
# torch.save(features_, 'res_train_features_cut.pt')
# torch.save(labels_, 'res_train_labels_cut.pt')
# torch.save(pred_, 'res_predict_labels_cut.pt')

# # Load the saved features and labels
# train_features = torch.load('train_features.pt')
# train_labels = torch.load('train_labels.pt')
#
# # Group the features where train_labels == 1
# grouped_features = train_features[train_labels == 1]
#
# # Save the grouped features
# torch.save(grouped_features, 'grouped_features_label_1.pt')
#
# # Print the shape of the saved features to verify
# print(f"Saved {train_features.shape[0]} total features.")
# print(f"Grouped features with label 1: {grouped_features.shape[0]}")
