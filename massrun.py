import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import time
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
import pandas.api.types
from sklearn.metrics import roc_curve, auc, roc_auc_score
import torchvision
import timm


# params
FREEZE = True # whether or not to freeze the weights
EPOCHS = 20 # number of training epochs
DUPE = True
#TRAIN_DIR = 'C:\\Users\\rngki\\Downloads\\train_images_hair_removed_dullrazor\\'
TRAIN_CSV = 'C:\\Users\\rngki\\Downloads\\train_dataset_with_skin_tone.csv'
TRAIN_DIR = ['./ISIC_2024_Training_Input/', "C:\\Users\\rngki\\Downloads\\train_images_hair_removed_dullrazor\\"]
#TRAIN_CSV = 'C:\\Users\\rngki\\Downloads\\train-metadata.csv'
#TRAIN_CSV = 'C:\\Users\\rngki\\Downloads\\cleaned_styled_data\\cleaned_styled_data\\cleaned_augmented_data.csv'
#TEST_DIR = 'C:\\Users\\rngki\\Downloads\\test_images_hair_removed_dullrazor\\'
#TEST_CSV = 'C:\\Users\\rngki\\Downloads\\test_dataset_with_skin_tone.csv'

MODELS = ["deit","efficientnet", "resnet50", "coatnet", "resnet34"]

DATA_NAMES = ['ISIC_2024_Training_Input', 'train_images_hair_removed_dullrazor']
TEST_DIR = ['./ISIC_2024_Training_Input/', "./ISIC_2024_Training_Input/"]
TEST_CSV = 'C:\\Users\\rngki\\Downloads\\test_dataset_with_skin_tone.csv'

"""
2024 ISIC Challenge primary prize scoring metric

Given a list of binary labels, an associated list of prediction 
scores ranging from [0,1], this function produces, as a single value, 
the partial area under the receiver operating characteristic (pAUC) 
above a given true positive rate (TPR).
https://en.wikipedia.org/wiki/Partial_Area_Under_the_ROC_Curve.

(c) 2024 Nicholas R Kurtansky, MSKCC
"""



class ParticipantVisibleError(Exception):
    pass

def initialize_model(model_name, num_classes=1):
    model = None
    
    if model_name == "efficientnet":
        model = torchvision.models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet34":
        model = torchvision.models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "resnet50":
        model = torchvision.models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "coatnet":
        model = timm.create_model('coatnet_0_rw_224', pretrained=True)
        num_ftrs = model.head.fc.in_features
        model.head.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == "deit":
        model = timm.create_model('deit_base_patch16_224', pretrained=True)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError("Invalid model name")
    
    if FREEZE:
        for param in model.parameters():
            model.requires_grad = False
    
    return model.to(device)

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str, min_tpr: float=0.80) -> float:
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
    for col in solution.columns:
        if col != 'is_malignant':
            del solution[col]
    
    for col in submission.columns:
        if col != 'prediction':
            del submission[col]

    # check submission is numeric
    if not pandas.api.types.is_numeric_dtype(submission.values):
        raise ParticipantVisibleError('Submission target column must be numeric')

    # rescale the target. set 0s to 1s and 1s to 0s (since sklearn only has max_fpr)
    v_gt = abs(solution.values.ravel()-1)
    
    # flip the submissions to their compliments
    v_pred = -1.0*submission.values.ravel()

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

#     # Equivalent code that uses sklearn's roc_auc_score
#     v_gt = abs(np.asarray(solution.values)-1)
#     v_pred = np.array([1.0 - x for x in submission.values])
#     max_fpr = abs(1-min_tpr)
#     partial_auc_scaled = roc_auc_score(v_gt, v_pred, max_fpr=max_fpr)
#     # change scale from [0.5, 1.0] to [0.5 * max_fpr**2, max_fpr]
#     # https://math.stackexchange.com/questions/914823/shift-numbers-into-a-different-range
#     partial_auc = 0.5 * max_fpr**2 + (max_fpr - 0.5 * max_fpr**2) / (1.0 - 0.5) * (partial_auc_scaled - 0.5)
    
    return(partial_auc)

for dir in range(len(TRAIN_DIR)):

    USE_SPLIT = False

    MALIGNANT = 'target'
    # MALIGNANT = 'target'
    MALIG_IDX = 1
    # MALIG_IDX = 3

    if not TEST_DIR:
        TEST_DIR = TRAIN_DIR[dir]
        USE_SPLIT = True
    if not TEST_CSV:
        TEST_CSV = TRAIN_CSV
        USE_SPLIT = True

    def balance_classes(df, label_col=MALIGNANT):
        # Separate malignant and non-malignant samples
        malig_df = df[df[label_col] == 1]
        non_malig_df = df[df[label_col] == 0]

        # Get the minority count
        min_count = min(len(malig_df), len(non_malig_df))

        # Downsample both classes to min_count (or oversample malignant if needed)
        non_malig_df_balanced = non_malig_df.sample(n=min_count, random_state=42)
        malig_df_balanced = malig_df.sample(n=min_count, replace=True, random_state=42)  # oversample if needed

        # Concatenate back
        balanced_df = pd.concat([malig_df_balanced, non_malig_df_balanced]).sample(frac=1, random_state=42).reset_index(drop=True)
        return balanced_df

    if USE_SPLIT:
        df = pd.read_csv(TRAIN_CSV)
        #df = pd.read_csv('C:\\Users\\rngki\\Downloads\\cleaned_styled_data\\cleaned_styled_data\\cleaned_augmented_data.csv')
        print(f"num malignant: {sum(df[MALIGNANT])}")

        # downsample for time
        df = df.sample(frac=.1, random_state=42)

        train_df, val_df = train_test_split(df, test_size=0.2, stratify=df[MALIGNANT], random_state=42)
    else:
        train_df = pd.read_csv(TRAIN_CSV)
        val_df = pd.read_csv(TEST_CSV)

    if DUPE:
        malig_df = train_df[train_df[MALIGNANT] == 1]
        malig_df = pd.concat([malig_df]*100, ignore_index=True)

        train_df = pd.concat([train_df, malig_df], ignore_index=True)


    # Balance the training set
    train_df = balance_classes(train_df)

    train_df.to_csv('train_labels.csv', index=False)
    val_df.to_csv('val_labels.csv', index=False)

    print(f"Validation - Num malignant: {val_df[MALIGNANT].sum()}")
    print(f"Validation - Num benign: {len(val_df) - val_df[MALIGNANT].sum()}")
    print(f"Training - Num malignant: {train_df[MALIGNANT].sum()}")
    print(f"Training - Num benign: {len(train_df) - train_df[MALIGNANT].sum()}")

    class ISICDataset(Dataset):
        def __init__(self, csv_file, img_dir, non_malignant_transform=None, malignant_transform=None):
            self.df = pd.read_csv(csv_file)
            self.img_dir = img_dir
            self.non_malignant_transform = non_malignant_transform
            self.malignant_transform = malignant_transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            img_path = os.path.join(self.img_dir, self.df.iloc[idx, 0] + '.jpg')

            image = Image.open(img_path).convert('RGB')

            if self.df.iloc[idx, MALIG_IDX] == 0 and self.non_malignant_transform:
                image = self.non_malignant_transform(image)
            elif self.df.iloc[idx, MALIG_IDX] == 1 and self.malignant_transform:
                image = self.malignant_transform(image)

            return image, int(self.df.iloc[idx, MALIG_IDX]), self.df.iloc[idx, 0]



    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet means
                            std=[0.229, 0.224, 0.225])    # ImageNet stds
    ])

    malignant_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = ISICDataset(
        csv_file='val_labels.csv',
        img_dir=TEST_DIR[dir],
        non_malignant_transform=transform,
        malignant_transform=transform
    )

    train_dataset = ISICDataset(
        csv_file='train_labels.csv',
        img_dir=TRAIN_DIR[dir],
        non_malignant_transform=transform,
        malignant_transform=malignant_transform
    )

    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    print(f"Training set size: {len(dataloader.dataset)}")
    print(f"Validation set size: {len(valloader.dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {'cuda' if torch.cuda.is_available() else 'cpu'}")

    for model_name in MODELS:
        resnet = initialize_model(model_name=model_name, num_classes=1)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet.parameters()), lr=1e-4)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))

        # switch to this if we're doing more than malignant/not-malignant
        #criterion = nn.CrossEntropyLoss()

        resnet.to(device)

        with open(f"training_log_{model_name}_{DATA_NAMES[dir]}.csv", "w") as f:
            f.write("epoch,train_loss,val_loss,val_tn,val_fn,val_tp,val_fp,val_accuracy,val_auc,pauc,f1score\n")
            start_time = time.time()
            for epoch in range(EPOCHS):
                resnet.train()
                avgloss = 0.0
                print(f"EPOCH: {epoch + 1}")
                for images, labels, _ in dataloader:
                    images, labels = images.to(device), labels.float().unsqueeze(1).to(device)

                    optimizer.zero_grad()
                    outputs = resnet(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    avgloss += loss.item() * images.size(0)
                avgloss /= len(dataloader.dataset)

                resnet.eval()
                val_loss = 0.0
                total = 0
                false_negative = 0
                false_positive = 0
                true_negative = 0
                true_positive = 0
                all_labels = []
                all_probs = []  # Store probabilities instead of binary predictions
                image_ids_list = []  # Collect image IDs for submission

                with torch.no_grad():
                    for images, labels, image_ids in valloader:  # Ensure dataset returns image IDs
                        images = images.to(device)
                        labels = labels.to(device).float().unsqueeze(1)
                        
                        # Forward pass
                        outputs = resnet(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * images.size(0)

                        # Get probabilities (before thresholding)
                        probabilities = torch.sigmoid(outputs).cpu().numpy().flatten()
                        all_probs.extend(probabilities)
                        
                        # Get binary predictions for confusion matrix
                        preds = (torch.sigmoid(outputs) > 0.5).float()
                        preds = preds.cpu().numpy().flatten()
                        
                        # Track image IDs for submission
                        image_ids_list.extend(image_ids)  # Assuming image_ids are strings
                        
                        # Update confusion matrix
                        for p, l in zip(preds, labels.cpu().numpy()):
                            if p == 0 and l == 0:
                                true_negative += 1
                            elif p == 0 and l == 1:
                                false_negative += 1
                            elif p == 1 and l == 0:
                                false_positive += 1
                            elif p == 1 and l == 1:
                                true_positive += 1
                        total += labels.size(0)
                        all_labels.extend(labels.cpu().numpy().flatten())

                # Calculate metrics
                avg_val_loss = val_loss / len(valloader.dataset)
                accuracy = (true_positive + true_negative) / total
                p_auc = roc_auc_score(all_labels, all_probs)  # Use probabilities for AUC
                f1score = f1_score(all_labels, [1 if x > 0.5 else 0 for x in all_probs])

                # Create submission DataFrame
                submission_df = pd.DataFrame({
                    'isic_id': image_ids_list,
                    'prediction': all_probs
                })

                solution_df = pd.DataFrame({
                    'isic_id': image_ids_list,
                    'is_malignant': all_labels
                })

                # Calculate pAUC using the competition metric
                try:
                    pAUC = score(
                        solution=solution_df,
                        submission=submission_df,
                        row_id_column_name='isic_id'  # Must match your column name
                    )
                except ParticipantVisibleError as e:
                    print(f"Scoring Error: {e}")
                    pAUC = -1  # Handle invalid submissions

                if (epoch+1) % 5 == 0:
                    torch.save(resnet.state_dict(), f"resnet50_{epoch}_{DUPE}_{FREEZE}_pos_weight(2).pth")
                
                print(f"Epoch [{epoch+1}/{EPOCHS}] "
                    f"Train Loss: {avgloss:.4f} "
                    f"Val Loss: {avg_val_loss:.4f}\n"
                    f"Val TN: {true_negative} FN: {false_negative} \n"
                    f"TP: {true_positive} FP: {false_positive}\n"
                    f"Val Accuracy: {accuracy:.4f}\n"
                    f"AUC: {p_auc:.4f}\n"
                    f"pAUC: {pAUC:.4f}\n"
                    f"f1 score: {f1score:.4f}\n")  # Add pAUC to output'
                f.write(f"{epoch+1},{avgloss},{avg_val_loss},{true_negative},{false_negative},{true_positive},{false_positive},{accuracy},{p_auc},{pAUC},{f1score}\n")
            
            
        print(f"Training time: {time.time() - start_time:.2f}s")

        with open(f"scores_{model_name}_{DATA_NAMES[dir]}.csv", "w") as f:
            f.write("isic_id,prediction,target\n")
            for id, fscore, actual in zip(image_ids_list, all_probs, all_labels):
                f.write(f"{id},{fscore},{actual}\n")

        

        def plot_roc_and_pauc(y_true, y_scores, min_tpr=0.80):
            # Flip labels and predictions (based on your scoring function)
            v_gt = abs(np.asarray(y_true) - 1)
            v_pred = -1.0 * np.asarray(y_scores)

            # Calculate ROC curve
            fpr, tpr, thresholds = roc_curve(v_gt, v_pred)

            # Compute full AUC and partial AUC
            full_auc = auc(fpr, tpr)
            max_fpr = abs(1 - min_tpr)

            # Partial AUC calculation (manual interpolation for the cutoff point)
            stop = np.searchsorted(fpr, max_fpr, "right")
            x_interp = [fpr[stop - 1], fpr[stop]]
            y_interp = [tpr[stop - 1], tpr[stop]]
            
            # Interpolated TPR at max_fpr
            interp_tpr = np.interp(max_fpr, x_interp, y_interp)
            
            # Create partial ROC arrays up to max_fpr
            pauc_fpr = np.append(fpr[:stop], max_fpr)
            pauc_tpr = np.append(tpr[:stop], interp_tpr)
            
            partial_auc = auc(pauc_fpr, pauc_tpr)

            # Plotting
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"ROC curve (AUC = {full_auc:.4f})")
            
            # Shade pAUC region
            plt.fill_between(pauc_fpr, pauc_tpr, step="post", alpha=0.3, color="orange", label=f"pAUC = {partial_auc:.4f}")
            
            # Draw horizontal line at TPR = min_tpr
            plt.axhline(min_tpr, color='red', linestyle='--', label=f"Min TPR = {min_tpr}")

            # Draw vertical line at FPR = max_fpr
            plt.axvline(max_fpr, color='green', linestyle='--', label=f"Max FPR = {max_fpr:.2f}")

            plt.xlabel("False Positive Rate (FPR)")
            plt.ylabel("True Positive Rate (TPR)")
            plt.title("ROC Curve with pAUC Region Highlighted")
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.savefig(f"roc_curve_{model_name}_{DATA_NAMES[dir]}.png")

        # Example usage
        plot_roc_and_pauc(all_labels, all_probs)