import streamlit as st
import joblib
import pandas as pd
import os


@st.cache_resource
def load_assets():
    reg_model = joblib.load("catboost_reg_model.joblib")
    reg_cols = joblib.load("model_columns.joblib")


    clf_model = joblib.load("catboost_clf_model.joblib")
    clf_cols = joblib.load("clf_columns.joblib")


    try:
        scaler = joblib.load("scaler_reg.joblib")
    except:
        scaler = None

    return reg_model, reg_cols, clf_model, clf_cols, scaler


def main():

    st.set_page_config(page_title="Proiect ML", layout="wide")


    try:
        reg_model, reg_cols, clf_model, clf_cols, scaler = load_assets()
    except Exception as e:
        st.error(f"Eroare la încărcarea fișierelor .joblib: {e}")
        return


    st.sidebar.title("Navigare Proiect")
    pagina = st.sidebar.radio("Selectează secțiunea:", ["Pagina Principală", "Explorare Date"])

    st.sidebar.divider()


    tip_analiza = "Regresie"
    if pagina == "Pagina Principală":
        tip_analiza = st.sidebar.selectbox("Prezicerea",
                                           ["Ore de lucru (Regresie)", "Nivel Venit (Clasificare)"])

    st.sidebar.subheader("Date Utilizator")
    age = st.sidebar.number_input("Vârstă", 17, 90, 30)
    edu_num = st.sidebar.number_input("Ani educație (educational-num)", 1, 16, 10)
    gender = st.sidebar.selectbox("Gen", ["Male", "Female"])
    marital = st.sidebar.selectbox("Stare civilă",
                                   ["Married-civ-spouse", "Never-married", "Divorced", "Separated", "Widowed"])
    relationship = st.sidebar.selectbox("Relație", ["Husband", "Not-in-family", "Own-child", "Unmarried", "Wife",
                                                    "Other-relative"])



    if pagina == "Pagina Principală":
        if tip_analiza == "Ore de lucru (Regresie)":
            st.title("Predicție Ore de Lucru (Regresie)")
            st.write("Acest model estimează numărul de ore lucrate pe săptămână pe baza profilului tău.")

            if st.button("Calculează Orele"):
                try:

                    input_df = pd.DataFrame(0, index=[0], columns=reg_cols)
                    input_df['age'] = age
                    input_df['educational-num'] = edu_num


                    if f"gender_{gender}" in input_df.columns: input_df[f"gender_{gender}"] = 1
                    if f"marital-status_{marital}" in input_df.columns: input_df[f"marital-status_{marital}"] = 1
                    if f"relationship_{relationship}" in input_df.columns: input_df[f"relationship_{relationship}"] = 1


                    final_input = scaler.transform(input_df) if scaler else input_df

                    pred = reg_model.predict(final_input)
                    st.success(f"### Rezultat estimat: {pred[0]:.2f} ore/săptămână")
                except Exception as e:
                    st.error(f"Eroare la predicția de regresie: {e}")

        else:
            st.title("Predicție Nivel Venit (Clasificare)")
            st.write("Acest model determină dacă este probabil ca venitul să depășească pragul de 50.000$/an.")

            if st.button("Analizează Venitul"):
                try:

                    input_df = pd.DataFrame(0, index=[0], columns=clf_cols)
                    input_df['age'] = age
                    input_df['educational-num'] = edu_num


                    if f"gender_{gender}" in input_df.columns: input_df[f"gender_{gender}"] = 1
                    if f"marital-status_{marital}" in input_df.columns: input_df[f"marital-status_{marital}"] = 1
                    if f"relationship_{relationship}" in input_df.columns: input_df[f"relationship_{relationship}"] = 1

                    prediction = clf_model.predict(input_df)


                    if prediction[0] == 1 or str(prediction[0]) == ">50K":
                        st.balloons()
                        st.success("###Verdict: Venit RIDICAT (>50.000$/an)")
                    else:
                        st.warning("###Verdict: Venit SCĂZUT (<=50.000$/an)")
                except Exception as e:
                    st.error(f"Eroare la predicția de clasificare: {e}")

    elif pagina == "Explorare Date":
        st.title("Analiza Vizuală a Performanței")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Curbe Învățare - Regresie")
            if os.path.exists("learning_curves_regresie.png"):
                st.image("learning_curves_regresie.png")
            else:
                st.warning("Imaginea pentru regresie lipsește.")

        with col2:
            st.subheader("Curbe Învățare - Clasificare")
            if os.path.exists("learning_curves_clasificare.png"):
                st.image("learning_curves_clasificare.png")
            else:
                st.warning("Imaginea pentru clasificare lipsește.")


if __name__ == "__main__":
    main()