from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from xgboost import XGBClassifier
from preprocessing import DataPreprocessor

class ModelPreprocessor:
    def handle_invalid_values(self, df, rules):
        df = df.copy()

        for col, (lower, upper) in rules.items():
            if col in df.columns:
                df.loc[(df[col] < lower) | (df[col] > upper),col] = np.nan
        return df

    def get_transformer(self, x_train):
        num_features = (x_train.select_dtypes(include=['int64', 'float64']).columns.tolist())
        ordinal_features = ['Credit_Mix', 'Payment_of_Min_Amount']
        nominal_features = ['Month', 'Occupation', 'Payment_Behaviour']

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        nominal_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        ordinal_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(categories=[
                ['Bad', 'Standard', 'Good'],
                ['No', 'Yes']
            ]))
        ])

        return ColumnTransformer([
            ('num', numeric_transformer, num_features),
            ('nom', nominal_transformer, nominal_features),
            ('ord', ordinal_transformer, ordinal_features)
        ])

class CreditScoreTrainer:
    def __init__(self):
        self.data_preprocessor = DataPreprocessor()
        self.model_preprocessor = ModelPreprocessor()

        mlflow.set_experiment("Credit Score Classification")

    def run(self, data_path):
        print("\n--- Step 2: Training ---")
        df = pd.read_csv(data_path)
        df = self.data_preprocessor.transform(df)

        x = df.drop('Credit_Score', axis=1)
        y = df['Credit_Score']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

        rules = {
            'Age': (18, 100),
            'Num_Bank_Accounts': (0, 50),
            'Num_Credit_Card': (0, 50),
            'Interest_Rate': (0, 100),
            'Num_of_Loan': (0, 50),
            'Delay_from_due_date': (0, 100),
            'Num_of_Delayed_Payment': (0, 100),
            'Changed_Credit_Limit' : (0, 50),
            'Num_Credit_Inquiries': (0, 100),
            'Monthly_Balance': (-1000000, 1000000)
        }

        x_train = self.model_preprocessor.handle_invalid_values(x_train, rules)
        x_test = self.model_preprocessor.handle_invalid_values(x_test, rules)
        transformer = self.model_preprocessor.get_transformer(x_train)

        models = {
            'RandomForest': Pipeline([
                ('preprocessing', transformer),
                ('classifier', RandomForestClassifier(
                    n_estimators=300,
                    max_depth=40,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ))
            ]),

            'ExtraTrees': Pipeline([
                ('preprocessing', transformer),
                ('classifier', ExtraTreesClassifier(
                    n_estimators=250,
                    max_depth=18,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                ))
            ]),

            'XGBoost': Pipeline([
                ('preprocessing', transformer),
                ('classifier', XGBClassifier(
                    objective='multi:softprob',
                    n_estimators=700,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=1.0,
                    random_state=42,
                    eval_metric='mlogloss'
                ))
            ])
        }

        Path("artifacts").mkdir(exist_ok=True)
        trained_models = {}
        run_ids = {}

        for name, model in models.items():
            with mlflow.start_run(run_name=name) as run:
                classifier = model.named_steps['classifier']
                mlflow.log_params(classifier.get_params())

                model.fit(x_train, y_train)
                model_path = f'artifacts/{name}.pkl'

                joblib.dump(model, model_path, compress=('xz', 3))
                mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

                trained_models[name] = model
                run_ids[name] = run.info.run_id

        return trained_models, run_ids, x_test, y_test