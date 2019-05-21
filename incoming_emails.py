import sqlite3, csv
import pandas as pd

db = "new_db.db"    
con = sqlite3.connect(db)
cur = con.cursor()

# Creation of emails_main table from a csv file.
def incoming_emails(con, csv_file):    
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip()
    df.to_sql('temp_incoming_emails', con, if_exists='replace')
    con.commit()

def update():
    # Update folder_directory in emails table and delete temporary table for unsupervised learning
    cur.executescript("""INSERT INTO emails_main(message_id, date, user, email_from, email_to, email_cc, email_bcc, email_subject, email_message)
                        SELECT message_id, date, user, email_from, email_to, email_cc, email_bcc, email_subject, email_message
                        FROM temp_incoming_emails
                        WHERE message_id NOT IN (SELECT message_id FROM emails_main);

                        DROP TABLE temp_incoming_emails;""")
    con.commit()

if __name__ == "__main__":
    incoming_emails(con, 'all_kmann.csv')
    update()