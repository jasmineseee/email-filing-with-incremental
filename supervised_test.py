import pandas as pd
import numpy as np

import sqlite3
db = "new_db.db"
con = sqlite3.connect(db)
cur = con.cursor()

from sklearn.linear_model import SGDRegressor

import os.path
from os import path

from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

def main():
    emails = list(cur.execute("""SELECT (
                                    COALESCE(email_from, '') || ' ' ||
                                    COALESCE(email_to, '') || ' ' ||
                                    COALESCE(email_cc, '') || ' ' ||
                                    COALESCE(email_bcc, '') || ' ' ||
                                    COALESCE(email_subject, '') || ' ' ||
                                    COALESCE(email_message, '')
                                )
                                
                                FROM emails_main
                                WHERE folder_directory IS NOT NULL"""))

    emails = [item[0] for item in emails]

    emails_folder = list(cur.execute("""SELECT folder_directory 
                                        FROM emails_main
                                        WHERE folder_directory IS NOT NULL"""))

    emails_folder = [item[0] for item in emails_folder]

    emails_tobeprocessed = list(cur.execute("""SELECT (
                                                COALESCE(email_from, '') || ' ' ||
                                                COALESCE(email_to, '') || ' ' ||
                                                COALESCE(email_cc, '') || ' ' ||
                                                COALESCE(email_bcc, '') || ' ' ||
                                                COALESCE(email_subject, '') || ' ' ||
                                                COALESCE(email_message, '')
                                            )
                                            
                                            FROM emails_main
                                            WHERE folder_directory IS NULL"""))

    emails_tobeprocessed = [item[0] for item in emails_tobeprocessed]

    emails_tobeprocessed_messageid = list(cur.execute("""SELECT message_id
                                                         FROM emails_main
                                                         WHERE folder_directory IS NULL"""))

    emails_tobeprocessed_messageid = [item[0] for item in emails_tobeprocessed_messageid]

    emails_tobeprocessed_tuple = list(zip(emails_tobeprocessed_messageid, emails_tobeprocessed))

    X_train = emails
    y_train = emails_folder
    labelencoder = LabelEncoder()
    labelencoder.fit(emails_folder)
    labelencoder_dict = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))
    y_train = labelencoder.transform(emails_folder)

    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)

    X_train = vectorizer.transform(X_train)

    X_predict = [email[1] for email in emails_tobeprocessed_tuple]
    X_predict = vectorizer.transform(X_predict)

    # OneClassSVM - to weed out outliers
    from sklearn.svm import OneClassSVM
    oneclasssvm = OneClassSVM()
    oneclasssvm.fit(X_train)
    oneclass_preds = list(oneclasssvm.predict(X_predict))

    # Get indexes of outliers
    # Outliers = -1, Inliers = 1
    outliers_indexes = [i for i,x in enumerate(oneclass_preds) if x == -1]

    if len(oneclass_preds) != len(outliers_indexes):
        # Delete the outliers, in reverse order so it doesn't throw off the subsequent indexes
        for index in sorted(outliers_indexes, reverse=True):
            del emails_tobeprocessed_tuple[index]

        # New value for X_predict after deletion of outliers
        X_predict = [email[1] for email in emails_tobeprocessed_tuple]
        X_predict = vectorizer.transform(X_predict)

        if path.exists('supervised_model.pkl') == False:
            # SGDRegressor Model
            model = SGDRegressor(warm_start=True)
            model.partial_fit(X_train, y_train)
        else:
            model = joblib.load('supervised_model.pkl')
            model.partial_fit(X_train, y_train)

        folder_directory = list(model.predict(X_predict))
        folder_directory = list(labelencoder.inverse_transform(folder_directory))

        message_id, email = zip(*emails_tobeprocessed_tuple)

        supervised_temp_df = pd.DataFrame({'message_id': message_id, 'folder': folder_directory})

        # Create a temporary table to store the results of supervised learning
        supervised_temp_df.to_sql('supervised_temp', con, if_exists='replace')

        # Update folder_directory in emails table and delete temporary table for supervised learning
        cur.executescript("""UPDATE emails_main
                             SET folder_directory = (
                                SELECT folder 
                                FROM supervised_temp 
                                WHERE message_id = emails_main.message_id)
                             WHERE emails_main.folder_directory IS NULL;
                                
                            DROP TABLE supervised_temp;""")
        con.commit()

    joblib.dump(model, 'supervised_model.pkl')

if __name__ == '__main__':
    main()