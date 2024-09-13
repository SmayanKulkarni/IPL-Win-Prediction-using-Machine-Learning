import streamlit as st
import pickle
import pandas as pd
import sklearn

# Load the pre-trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# IPL teams and cities
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi', 'Chandigarh',
          'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth', 'Durban', 'Centurion',
          'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad',
          'Cuttack', 'Nagpur', 'Dharamsala', 'Visakhapatnam', 'Pune', 'Raipur',
          'Ranchi', 'Abu Dhabi', 'Sharjah', 'Mohali', 'Bengaluru']

# Add header image
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQW__LeJsnrSCy0twaCt-Mb8ew21k1ob41nZg&s" alt="IPL Win Predictor" style="width: 50%;">
        <p><strong>IPL Win Predictor</strong></p>
    </div>
    """, 
    unsafe_allow_html=True
)

st.title('IPL Win Predictor')

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))

with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target', min_value=0)

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=1.0)
with col5:
    wickets_out = st.number_input('Wickets out', min_value=0, max_value=10)

if st.button('Predict Probability'):
    if score > target:
        # If score is more than the target, batting team wins
        st.header("Match Over")
        st.subheader(f"{batting_team} wins!!")
    elif wickets_out == 10:
        # Display "Match Over" and winner based on win probabilities
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets_out
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # Creating input dataframe for both teams
        input_df_batting = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        input_df_bowling = pd.DataFrame({
            'batting_team': [bowling_team],
            'bowling_team': [batting_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Predicting win probabilities for both teams
        result_batting = pipe.predict_proba(input_df_batting)
        result_bowling = pipe.predict_proba(input_df_bowling)
        
        win_prob_batting = result_batting[0][1]
        win_prob_bowling = result_bowling[0][1]

        # Display "Match Over" and the winner
        if win_prob_batting > win_prob_bowling:
            st.header("Match Over")
            st.subheader(f"{batting_team} wins!!")
        else:
            st.header("Match Over")
            st.subheader(f"{bowling_team} wins!!")
    else:
        # If the match is still ongoing
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets_out
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # Creating input dataframe
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Predicting win probability
        result = pipe.predict_proba(input_df)
        loss_prob = result[0][0]
        win_prob = result[0][1]

        # Displaying results
        st.header(f"{batting_team} - {round(win_prob * 100)}%")
        st.header(f"{bowling_team} - {round(loss_prob * 100)}%")