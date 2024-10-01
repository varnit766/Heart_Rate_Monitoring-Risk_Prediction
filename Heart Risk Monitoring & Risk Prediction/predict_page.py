import streamlit as st
import pickle
import numpy as np
import base64





def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data["model"]



Diet = {
  "Unhealthy":2,
  "Average":0,
  "Healthy":1
}

Sex = {
  "Male":1,
  "Female":0
}

yes_no = {
  "YES":1,
  "NO":0
}

risk = {
    "0":"❤️ is 💪",
    "1":"❤️ is ☠️"
}

new_person = {'Age': 22, 
              'Cholesterol': 324, 
              'BP_systolic': 174, 
              'BP_diastolic': 98, 
              'Heart Rate': 72,
              'Diabetes': 1, 
              'Family History': 0, 
              'Smoking': 0, 
              'Obesity': 1,
              'Alcohol Consumption':0, 
              'Exercise Hours Per Week': 2.09,
              'Previous Heart Problems':1, 
              'Medication Use':1,
              'BMI': 28.27, 
              'Triglycerides': 587, 
              'Sleep Hours Per Day' : 4,
              'Sex': 0, 
              'Diet': 1
              }

def determine_lifestyle_changes(predict_type, dictionary):
    
    # print(dictionary,new_person,new_person['BMI'])
    lifestyle_changes = ["So"]
    if predict_type > 0:
        if dictionary['Smoking'] == 1:
            lifestyle_changes.append('quit smoking')
        if dictionary['BMI'] < 18.5:
            lifestyle_changes.append('gain weight')
        elif dictionary['BMI'] > 25:
            lifestyle_changes.append('lose weight')
        if dictionary['Exercise Hours Per Week'] < 1.25:
            lifestyle_changes.append('do more exercise')
        if dictionary['Diet'] == 0:
            lifestyle_changes.append('eat healthy food')
        if dictionary['Alcohol Consumption'] == 1:
            lifestyle_changes.append('try reducing alcohol')
        st.subheader(f"Heart attack risk: {predict_type[0]}")
        # print("Heart attack risk:", predict_type)
        for i in lifestyle_changes:
            # print(f"Please {i},")
            st.subheader(f" {i},")
        # print("This can reduce your heart rate risk.")
        if len(lifestyle_changes)!=0:
          st.subheader("This can reduce your heart attack risk.")
          st.subheader("Remember, your heart's condition is a reflection of the choices you make. By following this guidance, you will be on the path to a heart full of life, strength, and love.")


    if predict_type > 0.75:
        # print("You should consult a doctor immediately.")
        st.subheader("You should consult a doctor immediately.")
        # print("Heart attack risk:", predict_type)



def show_predict_page():
  
  

  st.title("Heart Attack Predictor")
  

  st.write("""### Enter your details in the below mentioned fields :""")



  Diet_data = st.selectbox("Diet", Diet)
  Diet_data_int = Diet[Diet_data]

  Sex_data = st.selectbox("Sex", Sex)
  Sex_data_int = Sex[Sex_data]

  Age = st.slider("Age", 0, 100, 21)
  Sleep_Hours_Per_Day = st.slider("Sleep Hours Per Day 😴", 0, 24, 7)
  Exercise_Hours_Per_Week = st.slider("Exercise Hours Per Week 💪", 0, 6*5, 6)


  Diabetes = st.selectbox("Diabetes", ['YES', 'NO'], 1)
  Diabetes_data = yes_no[Diabetes]

  # st.write("""### data """ ,Sex_data_int,type(Sex_data_int))


  Family_History = st.selectbox("Family History", ['YES', 'NO'], 1)
  Family_History_data = yes_no[Family_History]

  Smoking = st.selectbox("Smoking", ['YES', 'NO'], 1)
  Smoking_data = yes_no[Smoking]

  Obesity = st.selectbox("Obesity", ['YES', 'NO'], 1)
  Obesity_data = yes_no[Obesity]

  Alcohol_Consumption = st.selectbox("Alcohol Consumption", ['YES', 'NO'], 1)
  Alcohol_Consumption_data = yes_no[Alcohol_Consumption]

  Medication_Use = st.selectbox("Medication Use", ['YES', 'NO'], 1)
  Medication_Use_data = yes_no[Medication_Use]

  Previous_Heart_Problems = st.selectbox("Previous Heart Problems", ['YES', 'NO'], 1)
  Previous_Heart_Problems_data = yes_no[Previous_Heart_Problems]



  Cholesterol = st.text_input('Cholesterol ?', '190')

  BP_systolic = st.text_input('BP_systolic ?', '120')
  BP_diastolic = st.text_input('BP_diastolic ?', '80')
  Heart_Rate = st.text_input('Heart_Rate ?', '80')
  BMI_data = st.text_input('BMI ?', '22')
  Triglycerides = st.text_input('Triglycerides ?', '150')


  st.write("""### data """ ,)

  ok = st.button("Prediction")
  if ok:
    X = np.array([[Age, Cholesterol, BP_systolic, BP_diastolic, Heart_Rate,Diabetes_data, Family_History_data,Smoking_data, Obesity_data,
       Alcohol_Consumption_data, Exercise_Hours_Per_Week,
       Previous_Heart_Problems_data,Medication_Use_data,BMI_data, Triglycerides
       , Sleep_Hours_Per_Day,
       Sex_data_int, Diet_data_int]])
    # X[:, 0] = le_country.transform(X[:,0])
    # X[:, 1] = le_education.transform(X[:,1])
    X = X.astype(float)
    # print(X)

    prediction = model.predict(X)
    predict_type = model.predict_proba(X)[:, 1]
    # print(predict_type)

    new_person = {'Age':  X[:,0], 
              'Cholesterol': X[:,1], 
              'BP_systolic': X[:,2], 
              'BP_diastolic': X[:,3], 
              'Heart Rate': X[:,4],
              'Diabetes': X[:,5], 
              'Family History': X[:,6], 
              'Smoking': X[:,7], 
              'Obesity': X[:,8],
              'Alcohol Consumption':X[:,9], 
              'Exercise Hours Per Week': X[:,10],
              'Previous Heart Problems':X[:,11], 
              'Medication Use':X[:,12],
              'BMI': X[:,13], 
              'Triglycerides': X[:,14], 
              'Sleep Hours Per Day' : X[:,15],
              'Sex': X[:,16], 
              'Diet': X[:,17]
              }
    
    print(risk[prediction[0].astype(str)])




    st.subheader(f"I, the divine observer, have seen into the depths of your heart's health and the condition is {risk[prediction[0].astype(str)]}")
    st.subheader("I wish to share with you the wisdom and guidance needed to keep your heart in a state of well-being.")
    st.subheader("Your heart is a precious gift, the center of your vitality, and a symbol of life's beauty. To nurture and protect it, I offer these insights:")

    # # predict_type = model.predict_proba(X)[:, 1]
    # st.subheader(f"prediction {predict_type[0]}")
    determine_lifestyle_changes(predict_type, new_person)

    st.subheader("May your heart beat in rhythm with the universe, filling your life with joy and health.")




