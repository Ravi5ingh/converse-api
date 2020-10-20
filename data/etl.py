import sys
import utility.util as ut

def run(train_csv_file_name):
    """
    Implementation of the ETL pipeline
    :param train_csv_file_name: The name of the train csv file
    """

    train_df = ut.read_csv(train_csv_file_name)

    train_df = ut.one_hot_encode(train_df, 'intent')

    train_df.to_csv('train.csv', index=False)

def main():
    """
    Point of entry (Takes 1 argument)
    """

    if len(sys.argv) == 2:
        train_csv_file_name = sys.argv[1]

        run(train_csv_file_name)

    else:
        print('Please provide the training CSV file with the following columns: \'query\', \'intent\''\
              '\n\nExample: python -m data.etl chatbot_train.csv')

if __name__ == '__main__':
    main()