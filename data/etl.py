import sys

def run(train_csv_file_name, train_db_file_name, table_name):
    """
    Implementation of the ETL pipeline
    :param train_csv_file_name: The name of the train csv file
    :param train_db_file_name: The name of the output train db file
    :param table_name: The name of the table in the db file
    """
    pass

def main():
    """
    Point of entry (Takes 3 arguments)
    """

    if len(sys.argv) == 4:
        train_csv_file_name, train_db_file_name, table_name = sys.argv[1:]

    else:
        print('Please provide 3 arguments for the ETL process' \
              '\nArgument 1: Input csv file'\
              '\nArgument 2: Name of output db file'\
              '\nArgument 3: Name of table in db file'
              '\n\nExample: python -m data.etl chatbot_train.csv chatbot_train.db Queries')

if __name__ == '__main__':
    main()