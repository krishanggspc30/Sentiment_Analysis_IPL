import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="IPL Fan Sentiment Dashboard", layout="wide")
st.title("🏏 IPL Fan Sentiment Dashboard")

# -------------------------------
# Load Data + Model
# -------------------------------
@st.cache_data
def load_data():
    df_raw = pd.read_csv("Cleaned_IPL_Dataset2.csv")
    df_rankings = pd.read_csv("enhanced_team_rankings.csv")
    return df_raw, df_rankings

@st.cache_resource
def load_model():
    model = joblib.load("investment_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    return model, encoders

df_raw, df_rankings = load_data()
model, encoders = load_model()

# -------------------------------
# Section 1: Dataset Preview
# -------------------------------
st.subheader("📄 Survey Dataset Preview")
st.dataframe(df_raw.head(10))

# -------------------------------
# Section 2: Team Rankings
# -------------------------------
st.subheader("🏆 Predicted Final Team Rankings Based on Fan Sentiment")
st.dataframe(df_rankings[["Predicted Rank", "Team", "Team Sentiment Score"]])

# -------------------------------
# Section 3: Ranking Visualization
# -------------------------------
st.subheader("📊 Team Sentiment Score Chart")
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=df_rankings, x="Team", y="Team Sentiment Score", palette="coolwarm", ax=ax)
plt.xticks(rotation=45)
plt.title("Predicted Rankings Based on Fan Sentiment")
st.pyplot(fig)

# -------------------------------
# Section 4: Download CSV
# -------------------------------
st.subheader("⬇️ Download Enhanced Team Rankings")
csv = df_rankings.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "enhanced_team_rankings.csv", "text/csv")

# -------------------------------
# Section 5: Predict Investment Level
# -------------------------------
st.subheader("🧠 Predict Your Emotional Investment in IPL")

with st.form("predictor_form"):
    age_group = st.selectbox("Age Group", ["10-18", "19-25", "26-35", "36+"])
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    fav_team = st.selectbox("Favorite IPL Team", sorted(encoders["Favorite IPL Team:"].classes_))
    fav_player = st.selectbox("Favorite Player in IPL", sorted(encoders["Who is your favorite player in IPL?"].classes_))
    support_duration = st.selectbox("How long have you supported this team?", encoders["How long have you supported this team?"].classes_)
    argued_online = st.radio("Have you ever argued online about IPL?", ["Yes", "No"])
    attended_match = st.radio("Have you ever attended an IPL match in a stadium?", ["Yes", "No"])
    submit = st.form_submit_button("Predict")

    if submit:
        input_dict = {
            "Age Group": age_group,
            "Gender": gender,
            "Favorite IPL Team:": fav_team,
            "Who is your favorite player in IPL?": fav_player,
            "How long have you supported this team?": support_duration,
            "Have you ever argued with someone online about IPL?": argued_online,
            "Have you ever attended an IPL match in a stadium?": attended_match,
        }

        encoded_input = []
        for feature in input_dict:
            value = input_dict[feature]
            encoded_value = encoders[feature].transform([value])[0]
            encoded_input.append(encoded_value)

        prediction = model.predict([encoded_input])[0]
        st.success(f"🎯 Predicted Emotional Investment Level: **{prediction}** (1 = Not at all, 5 = Very High)")
