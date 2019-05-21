import sqlite3
db = "new_db.db"
con = sqlite3.connect(db)
cur = con.cursor()

# Delete rows from preprocessed_emails table if email has been put into a folder
def delete_from_preprocessed_emails():
    cur.execute("""DELETE FROM preprocessed_emails 
                   WHERE preprocessed_emails.message_id 
                   IN (
                   SELECT emails_main.message_id 
                   FROM emails_main 
                   WHERE emails_main.folder_directory is NOT NULL)""")

    con.commit()
    con.close()

def main():
    delete_from_preprocessed_emails()

if __name__ == '__main__':
    main()