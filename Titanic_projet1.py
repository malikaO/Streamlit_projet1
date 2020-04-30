import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

st.title("Est-ce que vous auriez survécu au TITANIC?")
st.markdown("En fonction de votre âge, de votre sexe et de la classe que vous auriez choisi on va pouvoir prédire si vous auriez eu des chances de survivre au naufrage du Titanic") 


Fichier= "C:/Users/ouizb/Downloads/MACHING LEARNING/ALGO SUPERVISE/train.csv"

#@st.cache # si je mets en cahe les changements ne se voient plus sur la page
#def load_data():
df = pd.read_csv(Fichier)
# je agrde les colonnes dont j'ai besoin
df = df[['Survived', 'Pclass', 'Sex','Age']]
# on remplace par l'age moyen
df = df.fillna(df['Age'].mean()) 
# j'enocde mes valeurs quali
df['Sex'] = df['Sex'].map({'male':0, 'female':1})
#return df



#df = load_data()

# mon target
Y = df["Survived"]
# mes features
X = df.drop('Survived', axis=1)


sexe = st.sidebar.radio("Sélectionnez votre sexe.", ("Homme", "Femme"))

age = st.sidebar.slider("Sélectionnez votre âge", 0, 100 )

classe  = st.sidebar.radio("Quelle classe choisissez-vous pour votre cabine?", ("Classe 1", "Classe 2", "Classe 3"))

classe_encoder = {"Classe 1": 1, "Classe 2":2, "Classe 3": 3}


## alo random forest

model = RandomForestClassifier(n_estimators=100) # plein de params 
model.fit(X, Y) # faire le fit avec rfc
model_pred = model.predict(X)



st.warning("Modèle d'apprentissage utilisé: Random Tree Forest")
if sexe == 'Homme':
	st.write("Vous êtes un homme.")
	sexe_code = 0
else:
	st.write("Vous êtes une femme.")
	sexe_code = 1

st.write("Vous avez choisit la classe {}.".format(classe))

st.write("Vous avez {} ans.".format(age))


personne = [[classe_encoder[classe], sexe_code, age]]
survie = model.predict(personne)
chance_survie = model.predict_proba(personne)
chance_de_survivre = chance_survie

if survie[0] == 1:
	st.write("Auriez-vous survécu? Oui")
else:
	st.write("Auriez-vous survécu? Non")


st.write("Quelles auraient été vos chances de survivre: {}%.".format(chance_survie[0,1]*100))