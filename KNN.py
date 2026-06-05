import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier

def main():
    # Load Data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    prediction = pd.read_csv('prediction.csv')

    print('Train:', train)
    print('Test:', test)
    train.head()
    train.info()
    train.isnull().sum()
    test.isnull().sum()

    # Preprocessing
    train = train.drop(columns=["Id", "", ""])  # Drop Unnecessary Columns
    test = test.drop(columns=["Id", "", ""])    # Drop Unnecessary Columns

    cols_to_fill = ["", "", "", "", "", "", ""]  # Columns to be used for Model Training

    for col in cols_to_fill:
        mean_value = train[col].mean()
        train[col] = train[col].fillna(mean_value)
        test[col] = test[col].fillna(mean_value)

    # Label Encoding
    label_encoders = {}

    for c in ["", "", ""]:  # Categorical Columns to be used for Model Training
        label_encoders[c] = LabelEncoder()
        label_encoders[c].fit(pd.concat([train[c], test[c]]).astype(str))

        train[c] = label_encoders[c].transform(train[c].astype(str))
        test[c] = label_encoders[c].transform(test[c].astype(str))

    train.head()

    # Feature Engineering

    # Separate features and target
    X = train.drop(columns=["Predicted_Column"])  # Target Column to be used for Model Training
    y = train["Predicted_Column"]

    X_test = test.copy()

    # 5-Fold Stratified Cross Validation
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    fold_auc_scores = []
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    plt.figure(figsize=(8, 6))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(f"Train Shape: {X_train.shape}")
        print(f"Valid Shape: {X_val.shape}")

        # KNN Model
        model = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='minkowski',
            p=2
        )

        # Fit the model
        model.fit(
            X_train, 
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=25
            )

        # Evaluation
        val_pred = model.predict_proba(X_val)[:, 1]
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_auc_scores.append(fold_auc)
        
        print(f"Fold {fold} AUC: {fold_auc:.4f}")

        oof_preds[val_idx] = val_pred
        test_preds += (model.predict_proba(X_test)[:, 1] / skf.n_splits)

        # ROC Curve for Fold
        fpr, tpr, _ = roc_curve(y_val, val_pred)
        plt.plot(fpr,tpr,lw=2,label=f'Fold {fold} (AUC={fold_auc:.4f})')

    # Performance Summary
    overall_auc = roc_auc_score(y, oof_preds)

    print(f"Mean CV AUC: {np.mean(fold_auc_scores):.4f}")
    print(f"Std CV AUC : {np.std(fold_auc_scores):.4f}")
    print(f"OOF AUC    : {overall_auc:.4f}")

    # ROC Plot
    plt.plot([0, 1],[0, 1],linestyle='--',color='black',label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('5-Fold Cross Validation ROC Curves')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("roc_curve_knn.png",dpi=300,bbox_inches='tight')
    plt.show()
    prediction["Prediction"] = test_preds
    prediction.to_csv("prediction.csv", index=False)

if __name__ == "__main__":
    main()