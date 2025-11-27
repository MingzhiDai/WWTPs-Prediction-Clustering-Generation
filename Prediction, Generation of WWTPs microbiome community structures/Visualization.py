# Plot heatmaps of the test set predictions of relative abundance for each OTU type for the original data and the generated data
def get_pred_list(data_source, OTU_type, numberOfIterations):
    assert data_source in ['OG', 'Gen'], "Data source must be either 'OG' or 'Gen'"
    if data_source == 'OG':
        OG_pred_list= []
        for i in range(0,numberOfIterations):
            iteration = i+1 # Files are numbered from 1 to 5, not 0 to 4
#             pred_filename = 'Pred OG Data' + '/' + OTU_type + '/' + 'Repeat_' + str(iteration) + '/' + 'Gabp_log.xlsx'
            pred_filename = 'F://博三文件//伦敦国王学院//学术交流汇报//AI for WWTP//DE-BP python//gabp(新)//成果图//New_resluts_2//Data Visualisation-Tom Vinestock//Pred OG Data' + '/' + OTU_type + '/' + 'Repeat_' + str(iteration) + '/' + 'Gabp_log.xlsx'
            OG_pred_list.append(pd.read_excel(pred_filename, sheet_name = 'Prediction_values', index_col = 0))
        return OG_pred_list
    elif data_source == 'Gen':
        Gen_pred_list= []
        for i in range(0,numberOfIterations):
            iteration = i+1 # Files are numbered from 1 to 5, not 0 to 4
            gen_filename = 'F://博三文件//伦敦国王学院//学术交流汇报//AI for WWTP//DE-BP python//gabp(新)//成果图//New_resluts_2//Data Visualisation-Tom Vinestock//Pred Gen Data' + '/' + OTU_type + '/' + 'Repeat_' + str(iteration) + '/' + 'Gabp_log.xlsx'
            Gen_pred_list.append(pd.read_excel(gen_filename, sheet_name = 'Prediction_values', index_col = 0))
        return Gen_pred_list

def get_median_pred(data_source, OTU_type, numberOfIterations, error_type = 'RMSE'):
    assert error_type == 'RMSE'
    assert data_source in ['OG', 'Gen'], "Data source must be either 'OG' or 'Gen'"
    pred_list = get_pred_list(data_source, OTU_type, numberOfIterations)
    pred_RMSE_error_list = [np.sqrt(np.mean((pred_list[i] - obs)**2)) for i in range(0,numberOfIterations)]
    print(pred_RMSE_error_list)
    pred_RMSE_median_index = pred_RMSE_error_list.index((np.percentile(pred_RMSE_error_list[0],50, interpolation='nearest')))
    return pred_list[pred_RMSE_median_index]

def get_obs(OTU_type):
    obs_filename = 'F://博三文件//伦敦国王学院//学术交流汇报//AI for WWTP//DE-BP python//gabp(新)//成果图//New_resluts_2//Data Visualisation-Tom Vinestock//Pred Gen Data' + '/' + OTU_type + '/' + 'Repeat_' + str(1) + '/' + 'Gabp_log.xlsx'
    return pd.read_excel(obs_filename, sheet_name = 'Observation_values', index_col = 0)

import numpy as np
import pandas as pd
print(np.shape(get_obs('Phylum')))
numberOfIterations = 5
# for OTU_type in ["Phylum", "Class", "Order"]:
OTU_type = "Phylum"
obs = get_obs(OTU_type)
for data_source in ['OG', 'Gen']:
    data_source_desc = 'Original' if data_source == 'OG' else 'Original & Generated'
    predData = get_median_pred(data_source, OTU_type, numberOfIterations)
    # print(np.mean(predData.values))
    plot_heatmap(predData, OTU_type, title = OTU_type + " Predictions based on " + data_source_desc + " Data")