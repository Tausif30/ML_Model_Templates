import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import minimize
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import category_encoders as ce


def main():
    # Load Data
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    prediction = pd.read_csv("prediction.csv")

    print("Train:", train.shape)
    print("Test :", test.shape)

    # Preprocessing
    train = train.drop(columns=["Id", "", ""])
    test = test.drop(columns=["Id", "", ""])
    cols_to_fill = [
        "", "", "", "", "", "", ""
    ]
    for col in cols_to_fill:
        mean_value = train[col].mean()
        train[col] = train[col].fillna(mean_value)
        test[col] = test[col].fillna(mean_value)

    # Label Encoding
    label_encoders = {}
    for c in ["", "", ""]:
        le = LabelEncoder()
        le.fit(
        pd.concat([train[c], test[c]]).astype(str))
        train[c] = le.transform(train[c].astype(str))
        test[c] = le.transform(test[c].astype(str))
        label_encoders[c] = le

    # Separate features and target
    X = train.drop(columns=["Predicted_Column"]) # Target Column to be used for Model Training
    y = train["Predicted_Column"]
    X_test = test.copy()
    
    # 5 Fold Stratified Cross Validation Setup
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

    # OOF Predictions
    xgb_oof = np.zeros(len(X))
    lgb_oof = np.zeros(len(X))
    cat_oof = np.zeros(len(X))

    # Test Predictions
    xgb_test_preds = np.zeros(len(X_test))
    lgb_test_preds = np.zeros(len(X_test))
    cat_test_preds = np.zeros(len(X_test))

    # Fold Scores
    xgb_fold_scores = []
    lgb_fold_scores = []
    cat_fold_scores = []

    # Cross Validation Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y),start=1):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        print(f"\nFold {fold}")
        print(f"Train Shape: {X_train.shape}")
        print(f"Valid Shape: {X_val.shape}")


        # Standard Scaling
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        # Models to Train (Train Multiple Models and Collect OOF Predictions)

        # Model 1: XGBoost
        model = xgb.XGBClassifier(
                n_estimators=300,
                learning_rate=0.001,
                max_depth=10,
                subsample=0.5,
                colsample_bytree=0.5,
                random_state=42,
                n_jobs=-1,
                eval_metric='auc',
                early_stopping_rounds=30,
                min_child_weight=1,
                gamma=0.1,
                reg_alpha=0,
                reg_lambda=1,
                scale_pos_weight=1,
                tree_method='exact',
                max_bin=256
            )
        # Fit the model with early stopping
        model.fit(
            X_train, 
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=25
            )
        print(f"XGBoost AUC : {xgb_auc:.4f}")

        # Model 2: LightGBM
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
        # Fit the model with early stopping
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=30,verbose=False)]
        )
        print(f"LightGBM AUC : {lgb_auc:.4f}")

        # Model 3: CatBoost
        cat_model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=3,
            random_strength=1,
            od_type='Iter',
            od_wait=50,
            verbose=0,
            random_seed=42
        )
        # Fit the model with early stopping
        cat_model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            use_best_model=True
        )        
        print(f"CatBoost AUC: {cat_auc:.4f}")

    # Individual Model OOF Scores
    xgb_auc = roc_auc_score(y, xgb_oof)
    lgb_auc = roc_auc_score(y, lgb_oof)
    cat_auc = roc_auc_score(y, cat_oof)

    print("\nIndividual Model OOF Scores")
    print(f"XGBoost  : {xgb_auc:.4f}")
    print(f"LightGBM  : {lgb_auc:.4f}")
    print(f"CatBoost : {cat_auc:.4f}")

    # Optimize Ensemble Weights
    def objective(weights):
        weights = np.clip(weights,0,None)

        weights = (weights /weights.sum())
        blend = (weights[0] * xgb_oof + weights[1] * lgb_oof + weights[2] * cat_oof)
        return -roc_auc_score(y,blend)

    result = minimize(objective, x0=[1/3, 1/3, 1/3], method='Nelder-Mead')

    weights = np.clip(result.x,0,None)
    weights = (weights / weights.sum())

    print("\nOptimized Weights")
    print(f"XGBoost  : {weights[0]:.4f}")
    print(f"LightGBM  : {weights[1]:.4f}")
    print(f"CatBoost : {weights[2]:.4f}")

    # Final Ensemble Predictions
    ensemble_oof = (weights[0] * xgb_oof + weights[1] * lgb_oof + weights[2] * cat_oof)
    ensemble_test = (weights[0] * xgb_test_preds + weights[1] * lgb_test_preds + weights[2] * cat_test_preds)
    ensemble_auc = roc_auc_score(y, ensemble_oof)

    print(f"\nEnsemble OOF AUC : {ensemble_auc:.4f}")
  
    # ROC Plot
    plt.figure(figsize=(8, 6))
    for name, preds in [
        ("XGBoost", xgb_oof),
        ("LightGBM", lgb_oof),
        ("CatBoost", cat_oof),
        ("Ensemble", ensemble_oof)
    ]:
        fpr, tpr, _ = roc_curve(y, preds)
        auc_score = roc_auc_score(y, preds)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_score:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Ensemble ROC Curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig( "roc_curve_ensemble.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Save Predictions
    prediction["Prediction"] = ensemble_test
    prediction.to_csv( "prediction_ensemble.csv", index=False)
    print("\nSaved: prediction_ensemble.csv")

if __name__ == "__main__":
    main()