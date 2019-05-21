import sqlite3, csv
from sqlite3 import Error
import pandas as pd

db = "new_db.db"    
con = sqlite3.connect(db)
cur = con.cursor()

def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
 
    return None

# Creation of emails_main table from a csv file.
def create_emails_table(con, csv_file):    
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    df.to_sql('emails_main', con, if_exists='fail')
    con.commit()

# Creation of emails_main table (empty, for filling up later on)
# def create_emails_table():
#     cur.execute("""CREATE TABLE emails_main(
#                    message_id TEXT PRIMARY KEY,
#                    date TEXT,
#                    user TEXT,
#                    email_from TEXT,
#                    email_to TEXT,
#                    email_cc TEXT,
#                    email_bcc TEXT,
#                    email_subject TEXT,
#                    email_message TEXT,
#                    folder_directory TEXT)""")

def add_folderdirectory_column():
    cur.execute("ALTER TABLE emails_main ADD COLUMN folder_directory TEXT")
    con.commit()

def create_preprocessed_emails_table():
    cur.execute("CREATE TABLE IF NOT EXISTS preprocessed_emails(message_id TEXT, preprocessed_text TEXT, UNIQUE(message_id))")
    con.commit()

def main(filename):
    create_connection(db)
    try:
        create_emails_table(con, filename)
        add_folderdirectory_column()
    except ValueError:
        pass

    create_preprocessed_emails_table()
    
if __name__ == '__main__':
    main(filename)

