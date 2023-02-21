import streamlit as st 
import pickle 
import seaborn as sns 
import matplotlib.pyplot as plt 
import pandas as pd 
from transformers import pipeline

password_guess = st.text_input('What is the Password?')
if password_guess != st.secrets['password']:
  st.stop()


penguin_df = pd.read_csv('penguins.csv')
rf_pickle = open('random_forest_penguin.pickle', 'rb')
map_pickle = open('output_penguin.pickle', 'rb')

rfc = pickle.load(rf_pickle)
unique_penguin_mapping = pickle.load(map_pickle)

st.write(unique_penguin_mapping)

island = st.selectbox('Penguin Island', options = ["Biscoe", "Dream", "Torgerson"])
sex = st.selectbox('Sex', options = ['Male', 'Female'])
bill_length = st.number_input("Bill Length (mm)", min_value=0)
bill_depth = st.number_input("Bill Depth (mm)", min_value=0)
flipper_length = st.number_input("Flipper Length (mm)", min_value=0)
body_mass = st.number_input("Body Mass (g)", min_value=0)
user_inputs = [island, sex, bill_length, bill_depth, flipper_length, body_mass]

st.write(f"The user inputs are {user_inputs}")

island_biscoe, island_dream, island_torgerson = 0,0,0
if island == 'Biscoe': 
    island_biscoe = 1
elif island == 'Dream': 
    island_dream = 1 
elif island == 'Torgerson': 
    island_torgerson = 1
sex_male, sex_female = 0,0
if sex == 'Male': 
    sex_male = 1
else: 
    sex_female = 1


st.write('unique_penguin_mapping', unique_penguin_mapping)

new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length,
                               body_mass, island_biscoe, island_dream,
                               island_torgerson, sex_female, sex_male]])

prediction_species = unique_penguin_mapping[new_prediction]
st.write(f"Here is the prediction : {prediction_species}")

st.write("Here are the features importances : ")
st.image('feature_importance.png', width = 400)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['bill_length_mm'],
                 hue=penguin_df['species'])
plt.axvline(bill_length)
plt.title('Bill Length by Species')
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['bill_depth_mm'],
                 hue=penguin_df['species'])
plt.axvline(bill_depth)
plt.title('Bill Depth by Species')
st.pyplot(ax)

fig, ax = plt.subplots()
ax = sns.displot(x=penguin_df['flipper_length_mm'],
                 hue=penguin_df['species'])
plt.axvline(flipper_length)
plt.title('Flipper Length by Species')
st.pyplot(ax)