import sys
import utility.util as ut

def run(train_csv_filename, etl_train_csv_filename):
    """
    Implementation of the ETL pipeline
    :param train_csv_filename: The name of the train csv file
    :param etl_train_csv_filename: The name of the output csv file after performing ETL
    """

    train_df = ut.read_csv(train_csv_filename)

    train_df = ut.one_hot_encode(train_df, 'intent')

    train_df.to_csv(etl_train_csv_filename, index=False)

def main():
    """
    Point of entry (Takes 2 argument)
    """

    if len(sys.argv) == 3:
        train_csv_filename, etl_train_csv_filename = sys.argv[1:]
        run(train_csv_filename, etl_train_csv_filename)
    else:
        print('Please provide 2 arguments for the ETL process' \
              '\nArgument 1: Input csv file'\
              '\nArgument 2: Name of output csv file'\
              '\n\nExample: python -m data.etl chatbot_train.csv chatbot_train.csv')

if __name__ == '__main__':
    main()