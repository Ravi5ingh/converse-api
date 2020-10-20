import sys

def main():
    """
    Entry point
    """

    if len(sys.argv) == 3:
        train_csv = sys.argv[1]

    else:
        print('Please provide the training CSV file'\
              '\n\nExample: python -m model.train train.csv')


if __name__ == '__main__':
    main()