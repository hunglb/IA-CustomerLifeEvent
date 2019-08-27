# Copyright 2017, 2018 IBM. IPLA licensed Sample Materials.
"""
Sample Materials, provided under license.
Licensed Materials - Property of IBM
© Copyright IBM Corp. 2019. All Rights Reserved.
US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
"""
import pandas as pd
from sklearn import model_selection
from sklearn.externals import joblib
import sys
import os, json
from collections import OrderedDict
import shap
from life_event_prep import LifeEventPrep


global models
global event_types
global project_path
models = {}
project_path = None


def init():
    global models
    global event_types
    global project_path
    
    # set project_path
    project_path = os.environ.get("DSX_PROJECT_DIR")
    
    # the life_event_prep.py scripts saves out the last user inputs used for prepping the data
    # import this dictionary and pass the variables to the prep function
    # this ensures that the inputs used for prepping the training data are the same as those used for prepping the scoring data
    user_inputs_dict = joblib.load(open(project_path + '/datasets/training_user_inputs.joblib', 'rb'))
    # convert the dictionary into all the variables required. The dictionary key becomes the variable name
    globals().update(user_inputs_dict)
    
    # define predictable event_types
    event_types = target_event_type_ids
    
    # load models for each event_type
    for event_type in event_types:
        
        model_name = event_type+"_Model"
        version = "latest"
        model_parent_path = project_path + "/models/" + model_name + "/"
        metadata_path = model_parent_path + "metadata.json"
    
        # fetch info from metadata.json
        with open(metadata_path) as data_file:
            meta_data = json.load(data_file)
    
        # if latest version, find latest version from  metadata.json
        if (version == "latest"):
            version = meta_data.get("latestModelVersion")
    
        # prepare model path using model name and version
        model_path = model_parent_path + str(version) + "/model"
    
        # load model
        model = joblib.load(open(model_path, 'rb'))
    
        # initialize the SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # save model and explainer to dictionary
        models[event_type] = (model, explainer)

def score(args):
    global models
    global event_types
    global project_path
    
    # parse input arguments
    dataset_name = args.get("dataset_name")
    cust_id     = args.get("cust_id")
    sc_end_date  = args.get("sc_end_date")
    
    # load event data for selected cust_id
    dataset_path = project_path + "/datasets/" + dataset_name
    input_df = pd.read_csv(dataset_path, parse_dates=['EVENT_DATE'], dayfirst=True)
    input_df = input_df[input_df['CUSTOMER_ID'].isin([cust_id])]
    
    
    # prep the scoring data
    scoring_prep = LifeEventPrep(event_types, train_or_score='score', 
                                 scoring_end_date=sc_end_date, forecast_horizon=forecast_horizon, observation_window=observation_window, 
                                  life_event_minimum_target_count=life_event_minimum_target_count, norepeat_months=norepeat_months)
    prepped_data = scoring_prep.prep_data(input_df, 'score')
    
    result = {}
    for event_type, (model, explainer) in models.items():
        
        # handle empty data
        if prepped_data[event_type].shape[0] == 0:
            print("Data prep filtered out customer data. Unable to score.", file=sys.stderr)
            return None
    
        # predict on prepped data
        predictions = model.predict(prepped_data[event_type]).tolist()
        classes = None
        probabilities = None
        try:
            if hasattr(model, 'classes_'):
                classes = model.classes_.tolist()
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(prepped_data[event_type]).tolist()
        except:
            pass
        
        impact_features, shap_plot_html, data, elem_id = getImpactFeatures(explainer, prepped_data[event_type], str(type(model)))
        
        # add to result dictionary
        result[event_type] = {
            "classes": classes,
            "probabilities": probabilities,
            "predictions": predictions,
            "explain": impact_features,
            "explain_plot_html": shap_plot_html,
            "explain_plot_data": data,
            "explain_plot_elem_id": elem_id
        }

    return result


# function to extract highest impact features from SHAP as dictionary
def getImpactFeatures(explainer, df, model_type):
    shap_values = explainer.shap_values(df)
    
    # account for multi-output models
    shap_input = shap_values
    expected_input = explainer.expected_value
    if model_type.endswith("sklearn.ensemble.forest.RandomForestClassifier'>"):
        shap_input = shap_values[0]
        expected_input = explainer.expected_value[0]
    
    # Select top and bottom 3 impact features
    sv_row = pd.DataFrame(shap_input, columns=df.columns.tolist()).iloc[0].sort_values()
    neg_impact = sv_row.head(3)
    pos_impact = sv_row.tail(3)
    
    # Get HTML of shap plot
    plot = shap.force_plot(expected_input, shap_input, df.iloc[0,:]).data
    
    # Extract HTML and JS components
    script_split = plot.split('<script>')
    plot_html = script_split[0].strip('\n ')
    plot_js = script_split[1].split('</script>')[0]
    data = plot_js.split('SHAP.AdditiveForceVisualizer, ')[1].split('),')[0]
    elem_id = plot_js.split("document.getElementById('")[1].split("')")[0]
    
    impact_features = neg_impact.append(pos_impact)
    return(impact_features.to_dict(), plot_html, data, elem_id)


def test_score(args):
    """Call this method to score in development."""
    init()
    return score(args)


# test scoring as a job
# import pprint
# pprint.pprint(test_score({
#     'dataset_name': 'event.csv',
#     'cust_id': 1039,
#     'sc_end_date': '2018-08-31'
# }))
