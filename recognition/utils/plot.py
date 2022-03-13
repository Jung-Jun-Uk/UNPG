import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = plt.rcParams['font.serif']
# from menpo.visualize.viewmatplotlib import sample_colours_from_colourmap
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(score_dict, labels, methods, name, save_name):    
    #x_labels = [10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1]  
    #x_labels = [6 * 10 ** -3, 10 ** -3, 10 ** -2, 10 ** -1]  
    # colours = dict(zip(methods, sample_colours_from_colourmap(len(methods), 'Set2')))
    fig = plt.figure(figsize=(4,3))
    for i, method in enumerate(methods):
        fpr, tpr, _ = roc_curve(labels, score_dict[method])    
        roc_auc = auc(fpr, tpr)
        fpr = np.flipud(fpr)
        tpr = np.flipud(tpr)  # select largest tpr at same fpr        
        plt.plot(fpr,
                tpr,
                # color=colours[method],                
                lw=1.5,
                label=('%s(%0.2f%%)' %
                        (method, roc_auc * 100)))            
    #plt.xlim([10 ** -6, 0.1])
    #plt.ylim([0.3, 1.0])
    plt.grid(linestyle='--', linewidth=1)
    #plt.xticks(x_labels)
    #plt.yticks(np.linspace(0.3, 1.0, 8, endpoint=True))
    plt.xscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.legend(loc="lower right", fontsize='small')
    plt.savefig(save_name + '.png', bbox_inches="tight", dpi=300)
    

