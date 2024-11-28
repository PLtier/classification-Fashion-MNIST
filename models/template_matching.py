def template_matching(train_idx, test_idx, data, NUMBER_OF_CLASSES,cms,all_scores):
    import numpy as np
    import pandas as pd
    ######Get the different functions included in here
    clothes = data[train_idx]
    mean_templates = [np.mean(clothes[clothes[:,-1]==cloth_class],axis=0) for cloth_class in range(NUMBER_OF_CLASSES)]
    validation_clothes = data[test_idx]

    test_results = pd.DataFrame(test_idx)
    results_df = pd.DataFrame()

    for class_id in range(NUMBER_OF_CLASSES):
        euclidean_distance = np.linalg.norm(validation_clothes - mean_templates[class_id][np.newaxis, :], axis=1)
        results_df[class_id] = euclidean_distance

    test_results['Actual_score'] = validation_clothes[:,-1]
    test_results['Template_score'] = results_df.idxmin(axis=1)
                
    y_test = test_results['Actual_score']
    y_pred = test_results['Template_score']
    
    return y_test, y_pred