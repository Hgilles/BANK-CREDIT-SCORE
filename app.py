import streamlit as st 
import numpy as np
#from predicction import predict

import joblib 
rfc=joblib.load('rf_model.joblib')

st.header('BIENVENU SUR LA PAGE DE PREDICTION DE CLIENTS SOLVABLES!PROTOTYPE par HOUNME HOUETO EMERICK GILLES')

lang=st.radio('Dans quelle langue souhaitez-vous exécuter le programme?',('francais(par défaut)','English'))

if lang=='francais(par défaut)':
    Month=st.number_input('Entrez la durée du pret en mois: ')
    Age=st.number_input('Entrez l_age du client: ')
    Annual_Income=st.number_input('Entrez le salaire annuel du client: ')
    Monthly_Inhand_Salary=st.number_input('Entrez son salaire mensuel')
    Num_Bank_Accounts=st.number_input('Combien de comptes bancaires possède le client?: ')
    Num_Credit_Card=st.number_input('Combien de cartes bancaires a-t_il?: ')
    Interest_Rate=st.number_input('Quel est le taux d_intéret appliqué? ')
    Num_of_Loan=st.number_input('Combien de prets en cours possède t-il?: ')
    Num_of_Delayed_Payment=st.number_input('Entrez le montant de ses paiements mensuels: ')
    Num_Credit_Inquiries=st.number_input('Entrez le nombre de crédits constaté après enquete:')
    Credit_Mix=st.radio('Choissisez une appréciation pour son historique de crédit: ',('Bonne','Moyenne','Mauvaise'))
    Outstanding_Debt=st.number_input('Entrez la valeur estimée de ses autres dettes : ')
    Credit_Utilization_Ratio=st.number_input('Entrez la valeur de son ratio d_utilisation de credit: ')
    Credit_History_Age=st.number_input('Entrez la valeur en mois de la durée de ses crédits précedents: ')
    Total_EMI_per_month=st.number_input('Entrez la valeur mensuelle de son EMI:')
    Amount_invested_monthly=st.number_input('Entrez la valeur de ses investissements mensuels: ')
    Monthly_Balance=st.number_input('Entrez le solde mensuel de son compte: ')

    if Credit_Mix=='Bonne':
        Credit_Mix=2
    elif Credit_Mix=='Moyenne':
        Credit_Mix=1
    elif Credit_Mix=='Mauvaise':
        Credit_Mix=0

    x=np.array([Month,Age,Annual_Income,Monthly_Inhand_Salary,Num_Bank_Accounts,Num_Credit_Card,Interest_Rate,Num_of_Loan,Num_of_Delayed_Payment,Num_Credit_Inquiries,Credit_Mix,Outstanding_Debt,Credit_Utilization_Ratio,Credit_History_Age,Total_EMI_per_month,Amount_invested_monthly,Monthly_Balance]).reshape(1,17)
    y=rfc.predict(x)

    if(st.button('Prédire la solvabilité du client')):
        if y==0:
            st.warning('La prédiction est mauvaise, client presque insolvable!')
        elif y==1:
            st.text('La prédiction est moyenne pour ce client!')
        elif y==2:
            st.text('La prédiction est bonne pour ce client, Il est solvable!')
elif lang=="English":
    Month=st.number_input('Enter loan duration(in month): ')
    Age=st.number_input('Entrer Customer_s age: ')
    Annual_Income=st.number_input('Enter annual income: ')
    Monthly_Inhand_Salary=st.number_input('Enter monthly inhand salary')
    Num_Bank_Accounts=st.number_input('How many bank accounts does he has?: ')
    Num_Credit_Card=st.number_input('How many credit card does he has?: ')
    Interest_Rate=st.number_input('What_s the interest rate? ')
    Num_of_Loan=st.number_input('How many open loans does he has?: ')
    Num_of_Delayed_Payment=st.number_input('Enter his number of delayed payment: ')
    Num_Credit_Inquiries=st.number_input('Enter number of his credit inquiries:')
    Credit_Mix=st.radio('Choose his credit mix historic evaluation: ',('Good','Standard','Bad'))
    Outstanding_Debt=st.number_input('Enter his oustanding debt : ')
    Credit_Utilization_Ratio=st.number_input('Enter his credit utilization ratio: ')
    Credit_History_Age=st.number_input('Enter his credit history age(in month): ')
    Total_EMI_per_month=st.number_input('Enter his EMI_per_month:')
    Amount_invested_monthly=st.number_input('Enter his amount invested monthly: ')
    Monthly_Balance=st.number_input('Enter his monthly balance: ')

    if Credit_Mix=='Good':
        Credit_Mix=2
    elif Credit_Mix=='Standard':
        Credit_Mix=1
    elif Credit_Mix=='Bad':
        Credit_Mix=0

    x=np.array([Month,Age,Annual_Income,Monthly_Inhand_Salary,Num_Bank_Accounts,Num_Credit_Card,Interest_Rate,Num_of_Loan,Num_of_Delayed_Payment,Num_Credit_Inquiries,Credit_Mix,Outstanding_Debt,Credit_Utilization_Ratio,Credit_History_Age,Total_EMI_per_month,Amount_invested_monthly,Monthly_Balance]).reshape(1,17)
    y=rfc.predict(x)

    if(st.button('Predict')):
        if y==0:
            st.warning('The prediction is Bad, The Customer is not eligible!')
        elif y==1:
            st.text('The prediction is standard for this customer!')
        elif y==2:
            st.text('The prediction is Good, The Customer is eligible!')

