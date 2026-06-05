import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score

def main():
    # Load Data
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    predictions = pd.read_csv('predictions.csv')

    print('Train:', train)
    print('Test:', test)
    train.head()
    train.info()
    train.isnull().sum()
    test.isnull().sum()

    # Preprocessing
    train = train.drop(columns=["Id", "", ""]) # Drop Unnecessary Columns
    test = test.drop(columns=["Id", "", ""]) # Drop Unnecessary Columns
    cols_to_fill = ["", "", "", "", "", "", ""] # Columns to be used for Model Training
    for col in cols_to_fill:
        mean_value = train[col].mean()
        train[col] = train[col].fillna(mean_value)
        test[col] = test[col].fillna(mean_value)
    # Label Encoding
    label_encoders = {}
    for c in ["", "", ""]: # Categorical Columns to be used for Model Training
        label_encoders[c] = LabelEncoder()
        label_encoders[c].fit(pd.concat([train[c], test[c]]).astype(str))
        train[c] = label_encoders[c].transform(train[c].astype(str))
        test[c] = label_encoders[c].transform(test[c].astype(str))
    train.head()

    # Feature Engineering

    # Separate features and target
    X = train.drop(columns=["Prediction"])
    y = train["Prediction"]
    X_test = test.copy()

    # 5-Fold Stratified Cross Validation

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    fold_auc_scores = []
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    plt.figure(figsize=(8, 6))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        print(f"Train Shape: {X_train.shape}")
        print(f"Valid Shape: {X_val.shape}")

        model = lgb.LGBMClassifier(
            objective='binary',
            metric='auc',
            boosting_type='gbdt',
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=30,verbose=False)]
        )

        val_pred = model.predict_proba(X_val)[:, 1]

        fold_auc = roc_auc_score(y_val, val_pred)
        fold_auc_scores.append(fold_auc)

        print(f"Fold {fold} AUC: {fold_auc:.4f}")

        oof_preds[val_idx] = val_pred

        test_preds += (model.predict_proba(X_test)[:, 1]/ skf.n_splits)

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
    plt.savefig("roc_curve_lightgbm.png",dpi=300,bbox_inches='tight')
    plt.show()
    predictions["Prediction"] = test_preds
    predictions.to_csv("predictions.csv",index=False)

if __name__ == "__main__":
    main()