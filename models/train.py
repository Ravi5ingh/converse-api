import sys
import models.pipelines.ensemble_word2vec_classifier as ec
import utility.util as ut

def fit(train_csv):
    pass

def main():
    """
    Entry point
    """

    if len(sys.argv) == 3:
        train_csv, model_file_name = sys.argv[1:]

        classifier = ec.EnsembleWord2VecClassifier(train_csv)

        classifier.fit()

        ut.to_pkl(classifier, model_file_name)

    else:
        print('Please provide the training CSV file and the output model PKL file name'\
              '\n\nExample: python -m model.train train.csv model.pkl')


if __name__ == '__main__':
    main()