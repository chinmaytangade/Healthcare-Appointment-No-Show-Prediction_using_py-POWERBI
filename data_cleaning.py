# -----------------------------------------------------------
# data_cleaning.py
# -----------------------------------------------------------

import pandas as pd

df = pd.read_csv("KaggleV2-May-2016.csv")

df = df.drop(['PatientId', 'AppointmentID', 'Neighbourhood'], axis=1)

df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})

df = df[df['Age'] >= 0]

df['WaitingDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df = df[df['WaitingDays'] >= 0]

df.to_csv("cleaned_no_show_data.csv", index=False)
print("CLEANING COMPLETE → saved as cleaned_no_show_data.csv")
