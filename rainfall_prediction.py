import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

def main():
    # retrieve data
    url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
    df = pd.read_csv(url)
    df.head()

    # drop missing values
    df = df.dropna()

    # rename columns for clarification
    df = df.rename(columns={'RainToday': 'RainYesterday',
                            'RainTomorrow': 'RainToday'
                            })

    # use date_to_seasons function to transform dates to seasons
    df["Date"] = pd.to_datetime(df["Date"])
    df["Season"] = df["Date"].apply(date_to_season)
    df = df.drop(columns = "Date")

    # only pick locations around Melbourne
    df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])]

    # input and target data
    X = df.drop(columns = "Rainfall", axis = 1)
    y = df["Rainfall"]

    # transform "Rainfall" from continuous to binary
    y = y.where(y == 0.0, other = 1).astype(int)

    # split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)

    # split features into numeric and categorical, to apply OneHotEncoding and StandardScaler respectively
    numeric_features = X_train.select_dtypes(include = ["float64"]).columns.tolist()
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_features = X_train.select_dtypes(include = ["object","category"]).columns.tolist()
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # merge data again
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # create pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor",preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
        ]
    )

    # define parameters for GridSearch
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5]
    }

    # crossvalidation method to avoid data snooping
    cv = StratifiedKFold(n_splits=5, shuffle=True)

    # define gridsearch
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=cv, scoring="accuracy", verbose=2)
    grid_search.fit(X_train, y_train)

    # evaluate performance of Random Forest
    rf_best_params = grid_search.best_params_
    rf_best_score = grid_search.best_score_
    rf_test_score = grid_search.score(X_test, y_test)

    y_pred = grid_search.predict(X_test)
    rf_classification_report = classification_report(y_test, y_pred)
    rf_conf_matrix = confusion_matrix(y_test, y_pred)

    feature_names = numeric_features + list(grid_search.best_estimator_['preprocessor']
                                            .named_transformers_['cat']
                                            .named_steps['onehot']
                                            .get_feature_names_out(categorical_features))

    feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

    importance_df = pd.DataFrame({'Feature': feature_names,
                                'Importance': feature_importances
                                }).sort_values(by='Importance', ascending=False)

    N = 20  # Change this number to display more or fewer features
    rf_top_features = importance_df.head(N)


    ## train the model again, but with Logistic Regression this time
    # Replace RandomForestClassifier with LogisticRegression
    pipeline.set_params(classifier=LogisticRegression(random_state=42))

    # update the model's estimator to use the new pipeline
    grid_search.estimator = pipeline

    # Define a new grid with Logistic Regression parameters
    param_grid = {
        # 'classifier__n_estimators': [50, 100],
        # 'classifier__max_depth': [None, 10, 20],
        # 'classifier__min_samples_split': [2, 5],
        'classifier__solver' : ['liblinear'],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__class_weight' : [None, 'balanced']
    }

    grid_search.param_grid = param_grid

    # Fit the updated pipeline with LogisticRegression
    grid_search.fit(X_train, y_train)

    # Make predictions
    y_pred = grid_search.predict(X_test)

    # print performance of Random Forest
    print("\n### Performance Random Forest ###")
    
    print("\nBest parameters found: ", rf_best_params)
    print("Best cross-validation score: {:.2f}".format(rf_best_score))
    print("Test set score: {:.2f}".format(rf_test_score))

    print("\nClassification Report for Random Forest:")
    print(rf_classification_report)

    # plot confusion matrix of Random Forest
    disp = ConfusionMatrixDisplay(confusion_matrix=rf_conf_matrix)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix from Random Forest")

    # plot feature importance of Random Forest
    plt.figure(figsize=(10, 6))
    plt.barh(rf_top_features['Feature'], rf_top_features['Importance'], color='skyblue')
    plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature on top
    plt.title(f'Top {N} Most Important Features in predicting whether it will rain today from Random Forest')
    plt.xlabel('Importance Score')
    #plt.show()

    # print performance of Logistic Regression
    print("\n\n### Performance Logistic Regression ###")
    print("\nBest parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

    test_score = grid_search.score(X_test, y_test)
    print("Test set score: {:.2f}".format(test_score))

    y_pred = grid_search.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # plot confusion matrix for Logistic Regression
    conf_matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix from Logistic Regression")
    plt.show()


# map dates to the four seasons of the year (for more equal data points to learn from)
def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'
    

main()
