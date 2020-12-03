import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, auc, classification_report, plot_confusion_matrix, plot_roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# reads the data
df = pd.read_csv("data/heart_disease.csv")

# cleans up data
features = df.columns.to_numpy()[:-1]
data = df.iloc[:, :-1].to_numpy()

targets = df["target"].to_numpy()
targets = np.where(targets > 0, 1, targets)

def explore_decision_tree_depth():
    """Function calculates the training and test accuracy for different depths
    with the Decision Tree Classifier using cross validation. The results are plotted and saved.
    """   

    depths = np.arange(1, 50, 2)

    test_scores = np.zeros(len(depths))
    train_scores = np.zeros(len(depths))


    for i, depth in enumerate(depths):

        tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=1, criterion="gini")
        scores = cross_validate(estimator=tree, X=data, y=targets, cv=5, return_train_score=True)

        test_scores[i] = np.mean(scores["test_score"])
        train_scores[i] = np.mean(scores["train_score"])

    max_test = np.around(test_scores[np.argmax(test_scores)], 2)
    max_train = np.around(train_scores[np.argmax(train_scores)], 2)


    plt.plot(depths, test_scores, "-r", label="Test score - max = {}".format(max_test))
    plt.plot(depths, train_scores, "-b", label="Train score - max = {}".format(max_train))
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy score")
    plt.title("Decision Tree - Accuracy score")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("dt_max_depth_vs_accuracy.png", dpi=300)
    plt.show()


def explore_random_forests_estimators(max_features="log2"):
    """Function calculates the training and test accuracy for different number of estimators
    with the Random Forest Classifier using cross validation. The results are plotted and saved.
    """   

    scaler = StandardScaler(with_mean=True, with_std=False)
    scaler.fit(data)
    data_scaled = scaler.transform(data)

    estimators = np.arange(0, 1000, 10)
    estimators[0] = 1

    test_scores = np.zeros(len(estimators))
    train_scores = np.zeros(len(estimators))


    for j, n_estimators in enumerate(estimators):

        forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=10, min_samples_leaf=0.2, max_features=max_features)

        scores = cross_validate(estimator=forest, X=data_scaled, y=targets, cv=5, return_train_score=True)

        test_scores[j] = np.mean(scores["test_score"])
        train_scores[j] = np.mean(scores["train_score"])

        print("Explore random forests: n_estimators = {}".format(n_estimators))
        print("         Accuracy: test = {}, train = {}".format(test_scores[j], train_scores[j]))

    max_test = np.around(test_scores[np.argmax(test_scores)], 2)
    max_train = np.around(train_scores[np.argmax(train_scores)], 2)


    plt.plot(estimators, test_scores, "-r", label="test score - max = {}".format(max_test))
    plt.plot(estimators, train_scores, "-b", label="train score - max = {}".format(max_train))
    plt.xlabel("n_estimators")
    plt.ylabel("Accuracy score")
    plt.title("Random Forests - Accuracy score")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("rf_n_estimators_vs_accuracy.png", dpi=300)
    plt.show()

    
def explore_adaboost_learning_rates():
    """Function calculates the training and test accuracy for different learning rates
    with the AdaBoost Classifier using cross validation. The results are plotted and saved.
    """

    scaler = StandardScaler(with_mean=True, with_std=False)
    scaler.fit(data)
    data_scaled = scaler.transform(data)

    learning_rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

    test_scores = np.zeros(len(learning_rates))
    train_scores = np.zeros(len(learning_rates))


    for j, lr in enumerate(learning_rates):

        adaboost = AdaBoostClassifier(n_estimators=100, learning_rate=lr)

        scores = cross_validate(estimator=adaboost, X=data_scaled, y=targets, cv=5, return_train_score=True)

        test_scores[j] = np.mean(scores["test_score"])
        train_scores[j] = np.mean(scores["train_score"])

        print("Explore AdaBoost: learning_rate = {}".format(lr))
        print("         Accuracy: test = {}, train = {}".format(test_scores[j], train_scores[j]))

    max_test = np.around(test_scores[np.argmax(test_scores)], 2)
    max_train = np.around(train_scores[np.argmax(train_scores)], 2)


    plt.plot(np.arange(1, len(learning_rates) + 1), test_scores, "-r", label="test score - max = {}".format(max_test))
    plt.plot(np.arange(1, len(learning_rates) + 1), train_scores, "-b", label="train score - max = {}".format(max_train))
    plt.xlabel("learning_rate")
    plt.ylabel("Accuracy score")
    plt.xticks(np.arange(1, len(learning_rates) + 1), learning_rates)
    plt.title("AdaBoost - Accuracy score")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("abm_learning_rate_vs_accuracy.png", dpi=300)
    plt.show()

def explore_gradient_boosting_learning_rates():
    """Function calculates the training and test accuracy for different learning rates
    with the Gradient Boosting Classifier using cross validation. The results are plotted and saved.
    """

    scaler = StandardScaler(with_mean=True, with_std=False)
    scaler.fit(data)
    data_scaled = scaler.transform(data)

    learning_rates = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

    test_scores = np.zeros(len(learning_rates))
    train_scores = np.zeros(len(learning_rates))


    for j, lr in enumerate(learning_rates):

        gradboosting = GradientBoostingClassifier(n_estimators=50, learning_rate=lr, 
                                                  max_features="log2", max_depth=10,
                                                  min_samples_leaf=0.3)

        scores = cross_validate(estimator=gradboosting, X=data_scaled, y=targets, cv=5, return_train_score=True)

        test_scores[j] = np.mean(scores["test_score"])
        train_scores[j] = np.mean(scores["train_score"])

        print("Explore Gradient Boosting: learning_rate = {}".format(lr))
        print("         Accuracy: test = {}, train = {}".format(test_scores[j], train_scores[j]))

    max_test = np.around(test_scores[np.argmax(test_scores)], 2)
    max_train = np.around(train_scores[np.argmax(train_scores)], 2)


    plt.plot(np.arange(1, len(learning_rates) + 1), test_scores, "-r", label="test score - max = {}".format(max_test))
    plt.plot(np.arange(1, len(learning_rates) + 1), train_scores, "-b", label="train score - max = {}".format(max_train))
    plt.xlabel("learning_rate")
    plt.ylabel("Accuracy score")
    plt.xticks(np.arange(1, len(learning_rates) + 1), learning_rates)
    plt.title("Gradient Boosting - Accuracy score")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("gbm_learning_rate_vs_accuracy.png", dpi=300)
    plt.show()


def search_model(model):
    """The function calls grid_search() with the specified model and pre-defined parameters.

    Args:
        model (str): name of model to use, options are: [tree, forest, gradientboost, adaboost]
    """
    
    if model == "tree":
        params = {
            "min_samples_split":[2, 3, 4, 5],
            "min_samples_leaf":[1, 0.1, 0.2, 0.3, 0.4, 0.5],
            "max_depth":[5, 10, 20, 30, 40, 50, 100, None],
            "splitter":["best", "random"]
        }

        tree = DecisionTreeClassifier()
        grid_search(tree, params, "Decision Tree", "dt")


    elif model == "forest":
        params = {
            "n_estimators":[10, 50, 100, 250, 500, 750, 1000, 1500, 2000],
            "max_features":[None, "sqrt", "log2"],
            "max_depth":[None, 3, 5, 10, 20, 30, 40],
            "min_samples_leaf":[1, 0.1, 0.2, 0.3, 0.4, 0.5]
        }

        random_forest = RandomForestClassifier(criterion="gini")
        grid_search(random_forest, params, "Random Forests", "rf")

    elif model == "gradientboost":
        params = {
            "n_estimators":[10, 50, 100, 250, 500, 750, 1000, 1500, 2000],
            "learning_rate":[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
            "max_features":[None, "sqrt", "log2"],
            "max_depth":[None, 3, 5, 10, 20, 30, 40],
            "min_samples_leaf":[1, 0.1, 0.2, 0.3, 0.4, 0.5]
        }

        gradient_boosting = GradientBoostingClassifier()
        grid_search(gradient_boosting, params, "Gradient Boosting", "gbm")
    
    elif model == "adaboost":
        params = {
            "n_estimators":[10, 50, 100, 250, 500, 750, 1000, 1500, 2000],
            "learning_rate":[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
        }

        adaboost = AdaBoostClassifier()
        grid_search(adaboost, params, "AdaBoost", "abm")
    


def grid_search(model, params, name, suffix):
    """Performs a grid search of a collection of parameters with a specified classification model.
    
    The fucntion uses GridSeachCV from Scikit, which performs the grid search with 
    cross validation and can be parallelized.

    Writes the top 20 results to .txt-file in a latex table format, and also plots the 
    confusion matrix and roc curve with the best model from the grid search.

    Args:
        model (Object): model object to perform grid seach with
                        possible models are: DecisionTreeClassifier
                                             RandomForestClassifier
                                             GradientBoostingClassifier
                                             AdaBoostClassifier
        params (dict): a dictionary containg all parameters to test with model
        name (str): name of the model, used as title in plots
        suffix (str): shortened name for classification mehtod used in saved files and plots
    """
    
    # splits the datatset
    X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2)

    scaler = StandardScaler(with_mean=True, with_std=False)
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # performs the grid search
    clf = GridSearchCV(model, params, cv=5, n_jobs=-1, verbose=2)
    clf.fit(X_train, y_train)

    # tranforms dict to dataframe and sorts the results in order of performance
    frame = pd.DataFrame.from_dict(clf.cv_results_)
    frame = frame.sort_values("rank_test_score")

    # selects the relevant columns for the classification emthod
    columns = ["rank_test_score"]
    for param in params.keys():
        columns.append("param_" + param)

    columns.append("mean_test_score")

    # extracts the relevant columns and retrieves the top 10 results
    frame = frame[columns]
    frame = frame.iloc[:21, :]

    # writes results to txt file in latex table format
    with open(suffix + "_best_results_grid.txt", "w") as f:
        f.write(frame.to_latex(index=False))


    print(frame)
    print(clf.best_params_)
    print(clf.best_score_)

    # collects the parameters of the best performing model 
    str_params = ""
    for key, value in clf.best_params_.items():
        str_params += "_" + key + "=" + str(value) + ""


    # gets best model
    model = clf.best_estimator_

    # uses the best model to plot the confusion amtrix
    plot_confusion_matrix(model, X_test, y_test, display_labels=("Presence", "No presence"), normalize="true", cmap=plt.cm.Blues)
    plt.title(name + " - Confusion matrix")
    plt.savefig(suffix + "_confusion_matrix" + str_params + ".png", dpi=300)
    plt.show()

    # uses the best model to plot the roc curve with the auc score
    plot_roc_curve(model, X_test, y_test, name="ROC curve")
    plt.plot(np.arange(0.0, 1.1, 0.1), np.arange(0.0, 1.1, 0.1), "--k")
    plt.title(name + " - ROC/AUC on test set")
    plt.savefig(suffix + "_roc_curve" + str_params + ".png", dpi=300)
    plt.show()


if __name__ == "__main__":

    if len(sys.argv) == 3 and sys.argv[2] == "grid":
        search_model(sys.argv[1])

    elif len(sys.argv) == 2:
        
        if sys.argv[1] == "tree":
            explore_decision_tree_depth()
        elif sys.argv[1] == "forest":
            explore_random_forests_estimators()
        elif sys.argv[1] == "adaboost":
            explore_adaboost_learning_rates()
        elif sys.argv[1] == "gradientboost":
            explore_gradient_boosting_learning_rates()
        else:
            print("{} is not a valid classification method!".format(sys.argv[1]))
            
        
        
        
        
        