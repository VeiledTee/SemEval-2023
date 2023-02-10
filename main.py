from Embed import Embed
import pandas as pd
from XGBoostUncased import XGBoostUncased
from Ensemble import Ensemble
from ThresholdComparison import ThresholdComparison


def get_premise_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return pd.read_csv("Data/arguments-training.tsv", sep="\t"), pd.read_csv("Data/arguments-validation.tsv", sep="\t"), pd.read_csv("Data/arguments-test.tsv", sep="\t")


def get_label_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv("Data/labels-training.tsv", sep="\t"), pd.read_csv("Data/labels-validation.tsv", sep="\t")


if __name__ == '__main__':
    train, valid, test = get_premise_frames()
    labels_train, labels_valid = get_label_frames()

    train_embedding_uncased = Embed(train, 'uncased_training_tokens', True).saveEmbeddings()
    train_embedding_sentence = Embed(train, 'sentence_training_tokens', False).saveEmbeddings()
    valid_embedding_uncased = Embed(valid, 'uncased_validation_tokens', True).saveEmbeddings()
    valid_embedding_sentence = Embed(valid, 'sentence_validation_tokens', False).saveEmbeddings()
    test_embedding_uncased = Embed(test, 'uncased_testing_tokens', True).saveEmbeddings()
    test_embedding_sentence = Embed(test, 'sentence_testing_tokens', False).saveEmbeddings()

    # XGBoost model
    xgboost_parameters = {
        'objective': 'binary:logistic',
        'eta': 0.5,
        'max_depth': 15,
        'subsample': 0.25,
        'colsample_bytree': 1,
        'lambda': 0,
    }
    xgboost_model = XGBoostUncased(parameters=xgboost_parameters, test_frame=test, training_embeddings=None, testing_embeddings=None, training_file="JSON/uncased_training_tokens.json", testing_file="JSON/uncased_testing_tokens.json")
    xgboost_model.run(test_frame=test, labels=labels_train)

    # Ensemble Uncased
    e_uncased_parameters = {
            'multi_class': 'multinomial',
            'max_iter': 1000,
            'max_depth': 50,
            'n_estimators': 200,
            'voting': 'soft',
        }
    ensemble_uncased = Ensemble(parameters=e_uncased_parameters, test_frame=test, training_embeddings=None, testing_embeddings=None, training_file="JSON/uncased_training_tokens.json", testing_file="JSON/uncased_testing_tokens.json", uncased=True)
    ensemble_uncased.run(labels=labels_train)

    # Ensemble Sentence
    e_sentence_parameters = {
            'multi_class': 'multinomial',
            'max_iter': 100,
            'max_depth': 100,
            'n_estimators': 200,
            'voting': 'soft',
        }
    ensemble_sentence = Ensemble(parameters=e_sentence_parameters, test_frame=test, training_embeddings=None, testing_embeddings=None, training_file="JSON/sentence_training_tokens.json", testing_file="JSON/sentence_testing_tokens.json", uncased=False)
    ensemble_sentence.run(labels=labels_train)

    # Threshold Comparison
    unsupervised_comparison = ThresholdComparison(testing_embeddings=None, testing_file="JSON/sentence_testing_tokens.json")
    unsupervised_comparison.run(labels_train)
