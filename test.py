from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, recall_score, precision_score, roc_curve, auc, roc_auc_score, balanced_accuracy_score, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

model_folder = './models'
results_folder = './results'

def fetch_results(train=False):
    correct_drct = '_train' if (train==True) else '_eval'

    # Open files containing results and SVM model
    with open(model_folder + '/SVM.pkl', "rb") as f:
        model = pickle.load(f)

    with open(results_folder + '/bonafide' + correct_drct + '_scores.npy', "rb") as f:
        bonafide_results = np.load(f)

    with open(results_folder + '/spoofed' + correct_drct + '_scores.npy', "rb") as f:
        spoof_results = np.load(f)

    with open(results_folder + '/labels' + correct_drct + '.npy', "rb") as f:
        labels = np.load(f)


    train_X = np.stack((bonafide_results, spoof_results), axis=-1)
    train_Y = labels
    predictions = model.predict(train_X)
    probas = model.predict_proba(train_X)
    
    return (labels, predictions, probas, model)

# ROC Curve
def ROC_curve(labels, probas):
    lr_probs = probas[:, 1]
    lr_auc = roc_auc_score(labels, lr_probs)
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    lr_fpr, lr_tpr, _ = roc_curve(labels, lr_probs)
 
    plt.plot(lr_fpr, lr_tpr, color='darkorange', marker=',', label='ROC curve (area = %0.3f)' % lr_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='ROC curve (area = 0.5)')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

# Precision-Recall Curve
def PR_curve(labels, probas, predictions):
    probas = probas[:, 1]
    lr_precision, lr_recall, _ = precision_recall_curve(labels, probas)
    lr_f1, lr_auc = f1_score(labels, predictions), auc(lr_recall, lr_precision)
 
    plt.plot([0, 1], [0.1, 0.1], color='navy', linestyle='--')
    plt.plot(lr_recall, lr_precision, color='darkorange', marker=',', label='SVM: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='upper right')
    plt.show()

# Confusion Matrix
def Confusion_matrix(labels, predictions):
    cf_matrix = confusion_matrix(labels, predictions)
    group_counts = ['{0}'.format(value) for value in cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='magma')
    plt.show()
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='cividis')
    plt.show()

if __name__ == '__main__':
    # If true calculates metrics for 'train' dataset, else for 'dev'
    labels, predictions, probas, model = fetch_results(True)

    print('Accuracy: ', accuracy_score(labels, predictions))
    print('Balanced Accuracy: ', balanced_accuracy_score(labels, predictions))
    print('Recall: ', recall_score(labels, predictions))
    print('Precision: ', precision_score(labels, predictions))

    print(classification_report(labels, predictions))

    # ROC Curve
    ROC_curve(labels, probas)

    # Precision-Recall Curve
    PR_curve(labels, probas, predictions)

    # Confusion Matrix
    Confusion_matrix(labels, predictions)