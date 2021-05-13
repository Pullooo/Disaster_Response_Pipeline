import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(msg_df, cat_df):
    """
    Inputs:
    msg_df = input csv data containing the messages
    cat_df = input csv data containing the categories
    Output:
    returns df and categories representing merged data and categories data respectively
    """
    # load messages dataset
    messages = pd.read_csv(msg_df)
    # load categories dataset
    categories = pd.read_csv(cat_df)

    df = messages.merge(categories, how='inner', on='id')

    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = [field[0:-2] for field in row.values]

    # rename the columns of `categories`
    categories.columns = category_colnames

    return df, categories, category_colnames


def clean_data(colum_list, categories, main_df):
    """
    Cleans the input data
    """
    for column in colum_list:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    #     df.drop(columns = ['categories'], inplace = True)

    df2 = pd.concat([main_df, categories], axis=1)

    print("number of duplicates: {}".format(len(main_df.loc[main_df.duplicated()])))
    # drop duplicates
    df2.drop_duplicates(inplace=True)

    return df2


def save_data(input_data, db):
    """
    Save data to sqlite database
    Input:
    output_df: (str) output dataframe to be saved
    db_name: database name
    """
    engine_1 = create_engine("sqlite:///{}".format(db))
    input_data.to_sql('df_disaster', if_exists='replace', con=engine_1, index=False)


def main():
    if len(sys.argv) == 4:

        messages_df, cat_df, db = sys.argv[1:]

        try:
            print('Importing data sets...')
            temp_df, temp_cats, cols = load_data(messages_df, cat_df)

            print('Cleaning dataset.....')
            df2 = clean_data(cols, temp_cats, temp_df)

            print('Saving to database....')
            save_data(input_data=df2, db=db)
            print('load complete')
        except Exception as e:
            print(e)
            
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'messages.csv categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
