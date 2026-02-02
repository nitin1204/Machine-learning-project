import pandas as pd
import os

while True:
    df = pd.DataFrame({'studyhours':[input("enter study Hours:")],
                     'attendance':[input("enter attendance:")],
                     'pastscore':[input("enter pastscore:")],
                     'internet':[input("enter use internet(yes/no):")],
                     'sleephours':[input("enter sleephours:")],
                     'passed':[input("enter passed(yes/no):")]})
    

    file_name = "student_data1.csv"
    # data.to_csv('student_data.csv',index=False)
    if os.path.exists(file_name):
        df.to_csv(file_name, mode='a', header=False, index=False)
    else:
        df.to_csv(file_name, index=False)
   

    next_s =input("-------enter next student data(y/n)-------:").lower()
    if next_s != 'y':
        break
    print(df)
