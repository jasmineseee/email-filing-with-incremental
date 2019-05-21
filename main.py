import create_database, preprocessing, unsupervised, update_db, supervised_test, unsupervised_test

if __name__ == '__main__':
    create_database.main('500_kmann.csv')
    preprocessing.main()
    unsupervised_test.main()
    # supervised_test.main()
    update_db.main()