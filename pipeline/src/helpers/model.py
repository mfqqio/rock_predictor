import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict

# Calculates and shows the differences in the amount of explosive
# powder that would be loaded based on a given model's predictions.
def calc_explosives_cost(file_path, predictions, X, y, outfile):
    powd = pd.read_csv(file_path)
    y = y.to_frame()
    # import pdb; pdb.set_trace()
    # Add explosive powder amounts for target labels
    y_true_powder = pd.merge(y, powd,
                    how='left',
                    left_on='litho_rock_class',
                    right_on='rock_class')

    # Rename to signify these are powder amounts for true labels
    y_true_powder.rename(columns={'kg/m3':'kgm3_true',
                          'kg/t':'kgt_true',
                          'litho_rock_class':'true_rock_class'}, inplace=True)

    # Add explosive powder amounts for predicted labels
    pred_df = pd.DataFrame({'pred_rock_class':predictions})

    y_pred_powder = pd.merge(pred_df, powd,
                        how='left',
                        left_on='pred_rock_class',
                        right_on='rock_class')

    # Rename to signify these are powder amounts for predicted labels
    y_pred_powder.rename(columns={'kg/m3':'kgm3_pred',
                              'kg/t':'kgt_pred'}, inplace=True)
    # import pdb; pdb.set_trace()
    y_pred_powder.drop('rock_class', axis=1, inplace=True)

    # Combine target & predicted powder amounts together
    powders = pd.concat([y_true_powder, y_pred_powder], axis=1)

    # Calculate powder differences for each prediction
    powders['diff_m3'] =  powders['kgm3_true'] - powders['kgm3_pred']
    powders['abs_diff_m3'] = abs(powders['diff_m3'])

    # Sum of absolute difference in explosive powder between actual and predicted rock classes
    abs_diff = powders['abs_diff_m3'].sum()
    outfile.write('\n\nTotal absolute powder difference between actual and predicted rock classes: {:.4f} kg/m3'.format(abs_diff))
    under_blast = powders[powders['diff_m3'] > 0]['abs_diff_m3'].sum()
    over_blast = powders[powders['diff_m3'] < 0]['abs_diff_m3'].sum()

    outfile.write('\nOver-blast powder total: {:.4f} kg/m3'.format(over_blast))
    outfile.write('\nUnder-blast powder total: {:.4f} kg/m3'.format(under_blast))
    return

# For a given model, shows avg cross-validation score,
# confusion matrix and classification report of evaluation
# metrics including precision, recall, F1 score.
def evaluate(model, model_name, X, y, powder_path, outfile, cv_folds):
    predictions = model.predict(X)
    outfile.write('Evaluating {}...'.format(model_name))
    outfile.write('\nTrained on dataset of size {0} with {1} features\n'.format(X.shape, len(list(X))))
    acc = round(accuracy_score(y, predictions), 4)
    outfile.write('\nTrain accuracy: {}\n'.format(acc))

    # Calculates and shows F1 scores
    f1 = round(f1_score(y, predictions, average='macro'), 4)
    all_f1 = f1_score(y, predictions, average=None)
    outfile.write('\nAverage F1 Score: {}\n'.format(f1))

    all_f1 = f1_score(y, predictions, average=None)
    f1_classes = zip(model.classes_, all_f1)
    df_f1 = pd.DataFrame(list(f1_classes))
    df_f1.rename(columns={0:'litho_rock_class', 1:'F1_score'}, inplace=True)

    outfile.write('\n'+df_f1.to_string())

    calc_explosives_cost(powder_path, predictions, X, y, outfile)

    # Calculate and print cross validation score
    cv_scores = cross_val_score(model, X, y.values.ravel(), cv=cv_folds)
    mean_cv_score = np.mean(cv_scores)
    outfile.write("\n\nCross-validation accuracy ({0}-folds): {1:.4f}\n".format(cv_folds, mean_cv_score))

    # Create confusion matrix
    rock_labels = list(model.classes_)
    confus = confusion_matrix(y, predictions, labels=rock_labels)

    # Print confusion matrix with headers
    confus_ex = pd.DataFrame(confus,
                   index=['true:'+x for x in rock_labels],
                   columns=['pred:'+x for x in rock_labels])
    outfile.write('\nCONFUSION MATRIX\n {}'.format(confus_ex))

    # Classification report
    report = classification_report(y, predictions, target_names=rock_labels)
    outfile.write('\nCLASSIFICATION REPORT\n {}'.format(report))
    print('Model results written out to file.')
    return


# Runs cross validation for a specified number of folds and outputs an array of 
# predictions resulting from cross validation and probabilities for each class.
# Also prints out confusion matrix, classification report.
# Usage: run_cross_val_predict(clf, X, y, folds=5)
def run_cross_val_predict(model, X, y, folds):        
    y_pred = cross_val_predict(model, X, y.values.ravel(), cv=folds, method='predict_proba')
    
    # Format output and get most likely label
    probs = pd.DataFrame(y_pred, columns=[x for x in model.classes_])
    probs['y_pred'] = probs.idxmax(axis=1)
    
    # Print out eval metrics
    f1 = round(f1_score(y, probs['y_pred'], average='macro'), 4)
    print('\nAverage F1 Score:', f1)
    
    rock_labels = list(model.classes_)
    confus = confusion_matrix(y, probs['y_pred'], labels=rock_labels)

    # Print confusion matrix with headers
    confus_ex = pd.DataFrame(confus, 
                   index=['true:'+x for x in rock_labels], 
                   columns=['pred:'+x for x in rock_labels])
    print('\nCONFUSION MATRIX\n', confus_ex)
    
    # Classification report
    report = classification_report(y, probs['y_pred'], target_names=rock_labels)
    print('\nCLASSIFICATION REPORT\n', report)
    
    return probs