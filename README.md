🏏 IPL Fan Sentiment Analysis & Team Ranking Predictor
This project analyzes IPL fan sentiment data to predict team rankings and emotional investment levels using data mining, machine learning, and interactive dashboards.

📌 Project Highlights
🔍 Survey-based sentiment analysis from 400+ IPL fans

🧠 VADER NLP used to score fan reactions

📊 Exploratory Data Analysis with visualizations

🌟 Team sentiment score combining loyalty, optimism, and behavior

🤖 Random Forest model to predict emotional investment

🌐 Streamlit dashboard for live visualization and prediction

📁 Folder Structure
bash
Copy
Edit
ipl-fan-sentiment-project/
├── Cleaned_IPL_Dataset2.csv              # Cleaned and processed dataset
├── enhanced_team_rankings.csv            # Output file with predicted team scores
├── investment_model.pkl                  # Trained RandomForestClassifier
├── label_encoders.pkl                    # Encoders for input transformation
├── predicted_team_rankings.png           # Bar chart of team rankings (optional)
├── train_investment_model.py             # Script to train the prediction model
├── ipl_fan_dashboard.py                  # Streamlit dashboard app
└── README.md                             # Project summary and instructions
🔧 How It Works
📊 Sentiment Analysis
Reactions to wins/losses are scored using VADER

Scores are categorized as Positive / Neutral / Negative

🧠 Predictive Modeling
Fan behavior (age, loyalty, online activity) is used to train a model that predicts:

Emotional Investment Level (scale 1–5)

🏆 Team Ranking Prediction
Teams are scored using a weighted formula based on:

Sentiment

Fan loyalty

Optimism (believing the team will win)

Match attendance and discussions

Teams are ranked from 1 (most supported) to N

🌐 Streamlit Dashboard
Explore rankings

View dataset and bar charts

Predict your investment level by entering your fan profile

🖥️ How to Run Locally
✅ Prerequisites
Make sure you have Python 3.7+ installed.

📦 Install Requirements
bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit
▶️ Step 1: Train the Model
bash
Copy
Edit
python train_investment_model.py
This will create:

investment_model.pkl

label_encoders.pkl

▶️ Step 2: Launch Streamlit Dashboard
bash
Copy
Edit
streamlit run ipl_fan_dashboard.py
🖱️ Use the Dashboard to:
Preview the dataset

See predicted team rankings

Enter your fan profile and get your IPL emotional investment score

🚀 Future Enhancements
Integrate live Twitter sentiment

Add player-wise analysis

Deploy dashboard using Streamlit Cloud or Heroku
