import gensim
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
from utils.gensim_word2vec import GensimWord2VecVectorizer
from utils.utils import permutate_learner_parameters
from gensim.models import Word2Vec
from gensim.sklearn_api import W2VTransformer
from scipy import sparse
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.svm import LinearSVC, LinearSVR, NuSVC, NuSVR, OneClassSVM, SVC, SVR, l1_min_c
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report, confusion_matrix, \
                            average_precision_score, recall_score, f1_score, matthews_corrcoef, mean_absolute_error, \
                            ConfusionMatrixDisplay

class Metrics_Comparison(Enum):
    GREATER = 1
    LOWER = 2

class Model_Selection(Enum):
    TRAIN_TEST = 'TRAIN-TEST SPLIT'
    K_FOLD = 'K-FOLD'

data_settings = [
    {'model_selection': Model_Selection.TRAIN_TEST, 'test_size': 0.3},
    {'model_selection': Model_Selection.K_FOLD, 'n_folds': 10}
]

def get_train_test_split_arrays(X, Y, test_size=0.3):
    # Splits the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)

    # Formats the real values
    Y_true = np.array(Y_test).astype(int)

    # Transforms the training data into a 'document-term matrix'
    X_train_cv = get_bag_of_words_array(vocabulary_array=X_train, array_to_transform=X_train)

    # Transforms the testing data into a 'document-term matrix'
    X_test_cv = get_bag_of_words_array(vocabulary_array=X_train, array_to_transform=X_test)

    # Transforms the training data into a 'document-term matrix'
    X_train_tfidf = get_tfidf_array(vocabulary_array=X_train, array_to_transform=X_train)

    # Transforms the testing data into a 'document-term matrix'
    X_test_tfidf = get_tfidf_array(vocabulary_array=X_train, array_to_transform=X_test)

    # Transfroms the training data into a Word2Vec array
    X_train_w2v = sparse.csr_matrix(get_w2v_array(vocabulary_array=X_train,
                                                  array_to_transform=X_train,
                                                  Y=Y))

    # Transfroms the testing data into a Word2Vec array
    X_test_w2v = sparse.csr_matrix(get_w2v_array(vocabulary_array=X_train,
                                                  array_to_transform=X_test,
                                                  Y=Y))

    # Returns the information
    return {'general': {'X': X, 'Y': Y, 'X_train': X_train, 'X_test': X_test, 'Y_train': Y_train, 'Y_test': Y_test,
                        'train_size': 1 - test_size, 'test_size': test_size, 'Y_true': Y_true},
            'BoW': {'X_train': X_train_cv, 'X_test': X_test_cv}, 'Tf-idf': {'X_train': X_train_tfidf, 'X_test': X_test_tfidf},
            'Word2Vec': {'X_train': X_train_w2v, 'X_test': X_test_w2v}}

def get_k_folds_arrays(X, Y, n_folds):
    # Formats the real values
    Y_true = np.array(Y).astype(int)

    # Transforms the training data into a 'document-term matrix'
    X_train_cv = get_bag_of_words_array(vocabulary_array=X, array_to_transform=X)

    # Transforms the testing data into a 'document-term matrix'
    X_test_cv = get_bag_of_words_array(vocabulary_array=X, array_to_transform=X)

    # Transforms the training data into a 'document-term matrix'
    X_train_tfidf = get_tfidf_array(vocabulary_array=X, array_to_transform=X)

    # Transforms the testing data into a 'document-term matrix'
    X_test_tfidf = get_tfidf_array(vocabulary_array=X, array_to_transform=X)

    # Transfroms the training data into a Word2Vec array
    X_train_w2v = sparse.csr_matrix(get_w2v_array(vocabulary_array=X,
                                                  array_to_transform=X, Y=Y))

    # Transfroms the testing data into a Word2Vec array
    X_test_w2v = sparse.csr_matrix(get_w2v_array(vocabulary_array=X,
                                                  array_to_transform=X, Y=Y))

    # Returns the information
    return {'general': {'X': X, 'Y': Y, 'X_train': X, 'X_test': X, 'Y_train': Y, 'Y_test': Y, 'Y_true': Y_true,
                        'n_folds': n_folds},
            'BoW': {'X_train': X_train_cv, 'X_test': X_test_cv}, 'Tf-idf': {'X_train': X_train_tfidf, 'X_test': X_test_tfidf},
            'Word2Vec': {'X_train': X_train_w2v, 'X_test': X_test_w2v}}

def get_bag_of_words_array(vocabulary_array, array_to_transform):
    # Sets an instance for the Bag-of-Words
    cv = CountVectorizer()

    # Learns the vocabulary of the training data
    cv.fit(vocabulary_array)

    # Prints the vocabulary
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(vec.vocabulary_)

    # Returns the array transformed
    return cv.transform(array_to_transform)

def get_tfidf_array(vocabulary_array, array_to_transform):
    # Sets an instance for the Tfidf vectorizer
    tfidf = TfidfVectorizer(use_idf=True)

    # Learns the vocabulary of the training data
    tfidf.fit(vocabulary_array)

    # Returns the array transformed
    return tfidf.transform(array_to_transform)

def get_w2v_array(vocabulary_array, array_to_transform, Y):
    # Sets an instance for the Word2Vec
    gensim_word2vec_tr = GensimWord2VecVectorizer(size=50, min_count=3, sg=1,
                                                  alpha=0.025, iter=10)

    # Learns the vocabulary of the training data
    gensim_word2vec_tr.fit(vocabulary_array, Y)

    # Returns the array transformed
    return gensim_word2vec_tr.transform(array_to_transform)


def execute_a_series_of_naive_bayes_learnings(X, Y):
    # Sets a series of learnings
    learnings = [{'model': BernoulliNB(), 'model_text': 'Bernoulli Naive Bayes',
                  'params': dict(alpha=[1.0], binarize=[0.0], fit_prior=[True], class_prior=[None])},
                 {'model': GaussianNB(), 'model_text': 'Gaussian Naive Bayes',
                  'params': dict(priors=[None], var_smoothing=[1e-9])},
                 {'model': MultinomialNB(), 'model_text': 'Multinomial Naive Bayes',
                  'params': dict(alpha=[1.0], fit_prior=[True], class_prior=[None])}]

    # Executes the series of learnings
    execute_a_series_of_learnings(X=X, Y=Y, learnings=learnings)

def execute_a_series_of_support_vector_machine_learnings(X, Y):
    # Sets a series of learnings
    learnings = [{'model': SVC(), 'model_text': 'C-Support Vector Classification',
                  'params': dict(C=[1.0], kernel=['rbf', 'linear', 'poly', 'sigmoid', 'precomputed'], degree=[1, 2, 3, 4, 5],
                                 gamma=['scale', 'auto'], coef0=[0.0], shrinking=[True, False], probability=[False, True],
                                 tol=[1e-3], cache_size=[200], class_weight=[None], verbose=[False], max_iter=[-1],
                                 decision_function_shape=['ovr', 'ovo'], break_ties=[False, True], random_state=[None])}]

    # Executes the series of learnings
    execute_a_series_of_learnings(X=X, Y=Y, learnings=learnings)

def execute_a_series_of_random_forest_learnings(X, Y):
    # Sets a series of learnings
    learnings = [{'model': RandomForestClassifier(), 'model_text': 'Random Forest Classifier',
                  'params': dict(n_estimators=[75, 100, 125], criterion=['gini'], max_depth=[None],
                                 min_samples_split=[2], min_samples_leaf=[1], min_weight_fraction_leaf=[0.0],
                                 max_features=['auto', 'sqrt', 'log2'], max_leaf_nodes=[None], min_impurity_decrease=[0.0],
                                 bootstrap=[True], oob_score=[False], n_jobs=[None], random_state=[None],
                                 verbose=[0], warm_start=[False, True], class_weight=[None], ccp_alpha=[0.0],
                                 max_samples=[None])}]

    # Executes the series of learnings
    execute_a_series_of_learnings(X=X, Y=Y, learnings=learnings)

def execute_a_series_of_logistic_regression_learnings(X, Y):
    # Sets a series of learnings
    learnings = [{'model': LogisticRegression(), 'model_text': 'Logistic Regression',
                  'params': dict(penalty=['l2', 'l1', 'elasticnet', 'none'], dual=[False, True], tol=[1e-4], C=[1.0],
                                 fit_intercept=[True, False], intercept_scaling=[1], class_weight=[None], random_state=[None],
                                 solver=['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'], max_iter=[100],
                                 multi_class=['auto', 'ovr', 'multinomial'], verbose=[0], warm_start=[False], n_jobs=[None],
                                 l1_ratio=[None])}]

    # Executes the series of learnings
    execute_a_series_of_learnings(X=X, Y=Y, learnings=learnings)

def execute_a_series_of_k_nearest_neighbors_learnings(X, Y):
    # Sets a series of learnings
    learnings = [{'model': KNeighborsClassifier(), 'model_text': 'K-Nearest Neighbors',
                  'params': dict(n_neighbors=[1, 3, 5, 7, 10, 15], weights=['uniform', 'distance'],
                                 algorithm=['auto', 'ball_tree', 'kd_tree', 'brute'], leaf_size=[30], p=[2],
                                 metric=['minkowski'], metric_params=[None], n_jobs=[None])}]

    # Executes the series of learnings
    execute_a_series_of_learnings(X=X, Y=Y, learnings=learnings)

def execute_a_series_of_multi_layer_perceptron_learnings(X, Y):
    # Sets a series of learnings
    learnings = [{'model': MLPClassifier(), 'model_text': 'Multi-layer Perceptron',
                  'params': dict(hidden_layer_sizes=[(100,)], activation=['relu', 'identity', 'logistic', 'tanh'],
                                 solver=['adam', 'lbfgs', 'sgd'], alpha=[0.0001], batch_size=['auto'],
                                 learning_rate=['constant', 'invscaling', 'adaptive'], learning_rate_init=[0.001],
                                 power_t=[0.5], max_iter=[200], shuffle=[True, False], random_state=[None], tol=[1e-4],
                                 verbose=[False], warm_start=[False], momentum=[0.9], nesterovs_momentum=[True, False],
                                 early_stopping=[False, True], validation_fraction=[0.1], beta_1=[0.9], beta_2=[0.999],
                                 epsilon=[1e-8], n_iter_no_change=[10], max_fun=[15000])}]

    # Executes the series of learnings
    execute_a_series_of_learnings(X=X, Y=Y, learnings=learnings)

def execute_a_series_of_learnings(X, Y, learnings):
    # Iterates through the data settings
    for settings in data_settings:
        # New line character
        nl = '\n'

        # Prints the data settings
        print(f"\t--------------- {settings['model_selection'].value} ---------------{nl}")
        for key in settings:
            if key != 'model_selection':
                print(f'- {key}: {settings[key]}')

        # Gets the dataset information
        dataset = {}

        # Checks the data to get
        if settings['model_selection'] == Model_Selection.TRAIN_TEST:
            dataset = get_train_test_split_arrays(X=X, Y=Y, test_size=settings['test_size'])
        elif settings['model_selection'] == Model_Selection.K_FOLD:
            dataset = get_k_folds_arrays(X=X, Y=Y, n_folds=settings['n_folds'])

        # Iterates through the learnings to execute
        for learning in learnings:
            # Sets an array with the feature extractions
            feature_extractions = ['BoW', 'Tf-idf', 'Word2Vec']

            # Gets the permutations
            params = permutate_learner_parameters(params=learning['params'])

            # Iterates through the feature extractions
            for fe in feature_extractions:
                # Prints the feature extraction
                print(f'{nl}\t\t=============== {fe} ==============={nl}')

                # Executes the learner combinations
                execute_learner_combinations(model=learning['model'], model_text=learning['model_text'],
                                             params=params, dataset=dataset,
                                             model_selection_id=settings['model_selection'].value, feature_extraction_id=fe)

def execute_learner_combinations(model, model_text, params, dataset, model_selection_id, feature_extraction_id):    
    # Sets a list with the solutions
    solutions = []

    # Iterates through the permutations
    for item in params:
        # Sets a solution dictionary
        sol = {'params': item}

        try:
            # Executes the learning model, with the selected parameters
            results, params = execute_learner(model=model, params=item, dataset=dataset, model_selection_id=model_selection_id,
                                              feature_extraction_id=feature_extraction_id)

            # Updates the parameters
            sol['params'] = params

            # Appends the results
            sol['results'] = results
        except Exception as e:
            print(e)
            # Appends empty results
            sol['results'] = None

            # Notifies the user
            # print('Error! Something happened executing the learner')

        # Appends the solution
        solutions.append(sol)

    # Sets the dictionary with the keys
    metrics = {'accuracy': {'key': 'accuracy_score', 'condition': Metrics_Comparison.GREATER, 'current_best': None},
               'precision': {'key': 'precision_score', 'condition': Metrics_Comparison.GREATER, 'current_best': None},
               'recall': {'key': 'recall_score', 'condition': Metrics_Comparison.GREATER, 'current_best': None},
               'f1': {'key': 'f1_score', 'condition': Metrics_Comparison.GREATER, 'current_best': None},
               'mcc': {'key': 'mcc', 'condition': Metrics_Comparison.GREATER, 'current_best': None},
               'mae': {'key': 'mae', 'condition': Metrics_Comparison.LOWER, 'current_best': None}}

    # Iterates through the solutions
    for item in solutions:
        # Checks if the results exist
        if item['results'] is not None:
            # Iterates through the metrics
            for metric_id in metrics:
                # Gets the current metric
                metric = metrics[metric_id]

                # Checks if this is the best
                if metric['current_best'] is None:
                    metric['current_best'] = item
                else:
                    # Checks the comparison condition
                    if metric['condition'] == Metrics_Comparison.GREATER:
                        if item['results'][metric['key']] > metric['current_best']['results'][metric['key']]:
                            metric['current_best'] = item
                    elif metric['condition'] == Metrics_Comparison.LOWER:
                        if item['results'][metric['key']] < metric['current_best']['results'][metric['key']]:
                            metric['current_best'] = item

                # Updates the metric
                metrics[metric_id] = metric

    # Sets a list with the unique solutions
    unique = []

    # Iterates through the metrics
    for metric_id in metrics:
        # Checks if the current best is not None
        if metrics[metric_id]['current_best'] is not None:
            # Checks if the current best is in the list
            if metrics[metric_id]['current_best'] not in unique:
                unique.append(metrics[metric_id]['current_best'])

    # Sets a numeric ID
    i = 1

    # Iterates through the unique solutions
    for item in unique:
        # Gets the parameters
        params = item['params']

        # Gets the results
        results = item['results']

        # New line character
        nl = '\n'

        # Prints the current solution ID
        print(f'[#{i}]{nl}')

        # Prints the parameters
        print(f'* Parameters:{nl}')
        print(f'- Model: {model_text}')

        # Iterates through the params
        for param_id in params:
            print(f'- {param_id}: {params[param_id]}')

        # Prints the accuracy score
        print(f"{nl}Accuracy: {results['accuracy_score'] * 100} %{nl}")

        # Prints the average precision score
        print(f"Average precision score: {results['precision_score'] * 100} %{nl}")

        # Prints the average recall score
        print(f"Average recall score: {results['recall_score'] * 100} %{nl}")

        # Prints the average f-1 score
        print(f"Average F1-score: {results['f1_score'] * 100} %{nl}")

        # Prints the Matthews correlation coefficient
        print(f"Matthews correlation coefficient: {results['mcc']}{nl}")

        # Prints the mean absolute error
        print(f"Mean absolute error: {results['mae']}{nl}")

        # Prints the confusion matrix
        print(f'Confusion matrix:{nl}')
        disp = ConfusionMatrixDisplay(confusion_matrix=results['confusion_matrix'])
        disp = disp.plot(cmap=plt.cm.Blues,values_format='g')
        plt.show()

        # Prints the classification report
        print(f"{nl}Classification report:{nl}{nl}{results['classification_report']}{nl}")

        # Increases the numeric ID
        i += 1

def execute_learner(model, params, dataset, model_selection_id, feature_extraction_id):
    # Sets the parameters to the model
    model.set_params(**params)

    # Gets the parameters
    model_params = model.get_params()

    # Checks the model selection
    if model_selection_id == Model_Selection.TRAIN_TEST.value:
        # Fits the training data into the classifier
        clf = model.fit(dataset[feature_extraction_id]['X_train'].toarray(), dataset['general']['Y_train'])

        # Makes the prediction
        Y_pred = clf.predict(dataset[feature_extraction_id]['X_test'].toarray())
    elif model_selection_id == Model_Selection.K_FOLD.value:
        # Makes the prediction
        Y_pred = cross_val_predict(model, dataset[feature_extraction_id]['X_train'].toarray(), dataset['general']['Y_train'],
                                   cv=dataset['general']['n_folds'])

    # Gets the real values
    Y_true = dataset['general']['Y_true']

    # Formats the predicted values
    Y_pred = np.array(Y_pred).astype(int)

    return {'classification_report': classification_report(Y_true, Y_pred),
            'confusion_matrix': confusion_matrix(Y_true, Y_pred),
            'accuracy_score': accuracy_score(Y_true, Y_pred, normalize=True),
            'precision_score': average_precision_score(Y_true, Y_pred), 'recall_score': recall_score(Y_true, Y_pred,
                                                                                                     average='binary'),
            'f1_score': f1_score(Y_true, Y_pred, average='binary'), 'mcc': matthews_corrcoef(Y_true, Y_pred),
            'mae': mean_absolute_error(Y_true, Y_pred), 'Y_true': Y_true, 'Y_pred': Y_pred}, model_params