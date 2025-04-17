import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score

def filter_rows_by_target(input, target_column, comp_fn):
    filtered_df = input[comp_fn(input[target_column])]
    return filtered_df

def join_csv(df1, df2, column):
    merged_df = pd.merge(df1, df2, on=column, how='inner')
    return merged_df

# def get_frac(true_pos, false_neg, column, value):
#     true_ct = true_pos[column]
#     return f"{value}: {}"

def pAUC(input: pd.DataFrame, min_tpr: float=0.80) -> float:
    '''
    2024 ISIC Challenge metric: pAUC
    
    Given a solution file and submission file, this function returns the
    the partial area under the receiver operating characteristic (pAUC) 
    above a given true positive rate (TPR) = 0.80.
    https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve.
    
    (c) 2024 Nicholas R Kurtansky, MSKCC

    Args:
        solution: ground truth pd.DataFrame of 1s and 0s
        submission: solution dataframe of predictions of scores ranging [0, 1]

    Returns:
        Float value range [0, max_fpr]
    '''
    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    v_gt = abs(input["target"].values.ravel()-1)
    
    # flip the submissions to their compliments
    v_pred = -1.0*input["prediction"].values.ravel()

    max_fpr = abs(1-min_tpr)

    # using sklearn.metric functions: (1) roc_curve and (2) auc
    fpr, tpr, _ = roc_curve(v_gt, v_pred, sample_weight=None)
    if max_fpr is None or max_fpr == 1:
        return auc(fpr, tpr)
    if max_fpr <= 0 or max_fpr > 1:
        raise ValueError("Expected min_tpr in range [0, 1), got: %r" % min_tpr)
        
    # Add a single point at max_fpr by linear interpolation
    stop = np.searchsorted(fpr, max_fpr, "right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    partial_auc = auc(fpr, tpr)

    
    return(partial_auc)



# def plot(df, label):
#     # Create pivot table
#     pivot_table = df.pivot_table(index="iddx_3", columns="skin_tone", aggfunc="size", fill_value=0)
#     print(pivot_table)

#     # Ensure numeric values for plotting
#     pivot_table = pivot_table.apply(pd.to_numeric, errors='coerce').fillna(0)

#     # # Define the custom colors
#     custom_colors = ["#373028", "#422811", "#513B2E", "#6F503C", "#81654F", "#9D7A54", "#BEA07E", "#E5C8A6", "#E7C1B8", "#F3DAD6", "#FBF2F3"]
#     #
#     # # Plot stacked bar chart
#     # pivot_table.plot(kind="bar", stacked=True, figsize=(12, 6), cmap="tab10")
#     # pivot_table.plot(kind="bar", stacked=True, figsize=(12, 6), cmap=custom_colors)

#     # Generate Gradient Colors
#     start_color = "#513B2E"  # Darker
#     end_color = "#E7C1B8"    # Lighter

#     # Create gradient using linear interpolation
#     gradient_colors = list(mcolors.LinearSegmentedColormap.from_list("custom_gradient", [start_color, end_color])(np.linspace(0, 1, 7)))

#     # Plot stacked histogram
#     pivot_table.plot(kind="bar", stacked=True, figsize=(10, 6), color=gradient_colors)

#     # Wrap long x-axis labels
#     wrapped_labels = [textwrap.fill(label, width=15) for label in pivot_table.index]
#     print(wrapped_labels)
#     plt.xticks(ticks=range(len(wrapped_labels)), labels=wrapped_labels, rotation=0, ha="right")


#     # Formatting
#     plt.xlabel("Cancer Type")
#     plt.ylabel("Number of Cases")
#     plt.title(f"Distribution of Skin Tones Across Cancer Types ({label})")
#     plt.legend(title="Skin Tone", bbox_to_anchor=(1.05, 1), loc="upper left")
#     plt.xticks(rotation=45)
#     plt.tight_layout()

#     # Save plot
#     plt.savefig(f"{label}_skin_tones.png", dpi=300)

models = ["coatnet", "efficientnet", "resnet34", "resnet50", "deit"]
for model in models:
    print(model.upper())
    input = pd.read_csv(f'baseline/scores_{model}_ISIC_2024_Training_Input.csv')
    # input = pd.read_csv(f'dullrazor/scores_{model}_train_images_hair_removed_dullrazor.csv')
    # input_supp = pd.read_csv('../og_test_dataset_with_labeled_cancer_and_skin_tone.csv')[['isic_id', 'skin_tone']]
    input_supp = pd.read_csv('../ISIC_2024_Training_Supplement.csv')[['isic_id', 'iddx_full']]
    malignant = filter_rows_by_target(input, 'target', lambda x: x == 1.0)
    false_neg = filter_rows_by_target(malignant, 'prediction', lambda x: x < 0.5)
    true_pos = filter_rows_by_target(malignant, 'prediction', lambda x: x > 0.5)

    ### cancer type ###
    cancer_types = ["adnexal epithelial proliferations", "epidermal proliferations", "melanocytic proliferations"]
    # input_supp = pd.read_csv('../ISIC_2024_Training_Supplement.csv')
    # columns = ['iddx_full']
    # false_neg = join_csv(false_neg, input_supp, 'isic_id')[columns]
    # false_neg.to_csv('false_neg.csv', index=False)
    # true_pos = join_csv(true_pos, input_supp, 'isic_id')[columns]
    # true_pos.to_csv('true_pos.csv', index=False)

    ### skin tone ###
    skin_tones = ['#513B2E', '#6F503C', '#81654F', '#9D7A54', '#BEA07E', '#E5C8A6', '#E7C1B8']
    # columns = ['skin_tone']
    # false_neg = join_csv(false_neg, input_supp, 'isic_id')
    # true_pos = join_csv(true_pos, input_supp, 'isic_id')
    # print("True Positive:\n", true_pos['skin_tone'].value_counts())
    # print("False Negatives:\n", false_neg['skin_tone'].value_counts())

    ### confidence ###
    # input_supp = pd.read_csv('../ISIC_2024_Training_Supplement.csv')
    # columns = ['tbp_lv_dnn_lesion_confidence']
    # false_neg = join_csv(false_neg, input_supp, 'isic_id')[columns]
    # true_pos = join_csv(true_pos, input_supp, 'isic_id')[columns]
    # print("False negative:\n", false_neg.describe())
    # print("True positive:\n", true_pos.describe())

    ### confidence per skin tone ###
    # false_neg = join_csv(false_neg, input_supp, 'isic_id')
    # true_pos = join_csv(true_pos, input_supp, 'isic_id')
    # for skin_tone in skin_tones:
    #     fneg = filter_rows_by_target(false_neg, 'skin_tone', lambda x: x == skin_tone)[columns]
    #     tpos = filter_rows_by_target(true_pos, 'skin_tone', lambda x: x == skin_tone)[columns]
    #     print(f"False negative ({skin_tone}):\n", fneg.describe())
    #     print(f"True positive ({skin_tone}):\n", tpos.describe())

    merged = join_csv(input, input_supp, 'isic_id')
    # for skin_tone in skin_tones:
    #     df = filter_rows_by_target(merged, 'skin_tone', lambda x: x == skin_tone)
    for type in cancer_types:
        df = filter_rows_by_target(merged, 'iddx_full', lambda x: x.str.contains(type, case=False, na=False))
        predictions = (df["prediction"] >= 0.5).astype(int)
        # print('\t', skin_tone)
        # if skin_tone != "#513B2E":
        print('\t', type)
        if type != 'adnexal epithelial proliferations':
            print(f"\t\tpAUC: {pAUC(df)}")
            print(f"\t\tAUC: {roc_auc_score(df["target"], df["prediction"])}")
        print(f"\t\taccuracy: {accuracy_score(df["target"], predictions)}")
        print(f"\t\tbalanced accuracy: {balanced_accuracy_score(df["target"], predictions)}")
        print(f"\t\tf1_score: {f1_score(df["target"], predictions)}")
        