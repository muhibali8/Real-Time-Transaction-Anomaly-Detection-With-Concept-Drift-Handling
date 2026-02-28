import pandas as pd

import numpy as np

from datetime import datetime, timedelta
from collections import deque, defaultdict

import streamlit as st

import plotly.express as px
from river import anomaly, compose, preprocessing, drift


np.random.seed(42)

def transaction_stream( n_events=1000, start_time=datetime.now(), fraud_rate=0.02 ):
    current_time=start_time
    for i in range(n_events):
        # simulate time gap between transactions
        current_time += timedelta(seconds=np.random.exponential(scale=20))  # smaller gap for denser data scale=8

        # user behavior
        user_id = np.random.randint(1,50)   # fewer users to get more per-user data  -- 1, 15

        # DRIFT AFTER 500 EVENTS

        if i < 500:
            normal_mean = 50
            normal_std = 5
        else:
            normal_mean = 600      # NEW NORMAL
            normal_std = 5

        # fraud or not
        is_fraud = np.random.rand() < fraud_rate

        # transaction amount
        if is_fraud:
            amount= np.random.normal(loc=800, scale=150)
        else:
            amount= np.random.normal(loc=normal_mean, scale=normal_std)

        yield {
            'timestamp': current_time,
            'user_id': user_id,
            'amount': max(1, round(amount, 2)),
            'is_fraud': int(is_fraud)
        }

events=[]
for event in transaction_stream(n_events=1000):
    events.append(event)


window_all=deque()
user_windows= defaultdict(deque)


window_duration_5min = timedelta(minutes=5)    # global window
window_duration_10min = timedelta(minutes=10)  # per-user window


model= compose.Pipeline(
            preprocessing.StandardScaler(),
            anomaly.HalfSpaceTrees(n_trees=15, height=3, window_size=200, seed=42)
        )

#Drift detector
adwin= drift.ADWIN(delta=0.2)


threshold=0.68  #for testing
scores=[]
alerts=[]
events_with_features= []
drift_points= []

# Optional: pause learning after drift
pause_learning = 0
pause_after_drift = 50

# Streamlit setup

st.set_page_config(page_title="Fraud Detection Monitor", layout="wide")
st.title('Real-Time Transaction Anomaly Detection')
st.sidebar.header('Metrics')

for event in transaction_stream(n_events=1000):
    # Remove old events outside window
    while window_all and (event['timestamp'] - window_all[0]['timestamp']) > window_duration_5min:
        window_all.popleft()

    #Remove old events from per-user window
    user_id= event['user_id']
    while user_windows[user_id] and (event['timestamp'] - user_windows[user_id][0]['timestamp']) > window_duration_10min:
        user_windows[user_id].popleft()

    # Compute global rolling features

    if window_all:
        rolling_mean= np.mean([e['amount'] for e in window_all])
        rolling_max= np.max([e['amount'] for e in window_all])
        rolling_std= np.std([e['amount'] for e in window_all])
        rolling_count= len(window_all)
    else:
        rolling_mean = event['amount']
        rolling_max = event['amount']
        rolling_std= 0
        rolling_count = 1

    # Compute per-user rolling features

    if user_windows[user_id]:
        user_rolling_mean= np.mean([e['amount'] for e in user_windows[user_id]])
        user_rolling_count= len(user_windows[user_id])

    else:
        user_rolling_mean = event['amount']
        user_rolling_count = 1


    diff_from_global_mean= event['amount'] - rolling_mean
    diff_from_user_mean= event['amount'] - user_rolling_mean


    event['rolling_mean_5min'] = rolling_mean
    event['rolling_max_5min'] = rolling_max
    event['rolling_std_5min'] = rolling_std
    event['rolling_count_5min'] = rolling_count
    event['user_rolling_mean_10min'] = user_rolling_mean
    event['user_rolling_count_10min'] = user_rolling_count
    event['diff_from_global_mean'] = diff_from_global_mean
    event['diff_from_user_mean'] = diff_from_user_mean

    # Add current event AFTER computing feature

    events_with_features.append(event)




    #building features
    features ={
        'rolling_mean_5min': rolling_mean,
        'rolling_max_5min': rolling_max,
        'rolling_std_5min': rolling_std,
        'rolling_count_5min': rolling_count,
        'user_rolling_mean_10min': user_rolling_mean,
        'user_rolling_count_10min': user_rolling_count,
        'diff_from_global_mean': diff_from_global_mean,
        'diff_from_user_mean': diff_from_user_mean,
        'amount': event['amount']

    }


    score= model.score_one(features)
    scores.append(score)

    alert= score>threshold
    alerts.append(alert)


    # Drift Detection
    drift_detected = adwin.update(diff_from_global_mean)

    if adwin.drift_detected:
        drift_points.append(event['timestamp'])

        model= compose.Pipeline(
            preprocessing.StandardScaler(),
            anomaly.HalfSpaceTrees(n_trees=15, height=3, window_size=200, seed=42)
        )
        pause_learning = pause_after_drift  # optionally skip learning to let scores spike


    # Model learning
    if pause_learning > 0:
        pause_learning -= 1
    else:
        model.learn_one(features)


    window_all.append(event)
    user_windows[user_id].append(event)


# DataFrame for Streamlit

df_live = pd.DataFrame(events_with_features)
df_live['scores']= scores
df_live['alerts']= alerts


# Streamlit visualizations

#Metrics

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", len(df_live))
col2.metric("Total Alerts", sum(alerts))
col3.metric("Max Anomaly Score", np.round(max(scores),2))
col4.metric("Drift Detections:", len(drift_points))


st.sidebar.metric("Total Transactions", len(df_live))
st.sidebar.metric("Total Alerts:", sum(alerts))
st.sidebar.metric("Max Anomaly Score:", np.round(max(scores),2))
st.sidebar.metric("Drift Detections:", len(drift_points))


# Recent Transactions Table

st.subheader('Recent Transactions')
st.dataframe(df_live.tail(20))


# Color-Coded Alerts

latest = df_live.iloc[-1]

if latest['alerts'] == 1:
    st.write("Latest score:", latest["scores"])
    st.write("Alert value:", latest["alerts"])
    st.error(f" ALERT: User {latest['user_id']} | Amount: {latest['amount']} | Score: {np.round(latest['scores'],2)}")
else:
    st.write("Latest score:", latest["scores"])
    st.write("Alert value:", latest["alerts"])
    st.success("Latest transaction: Normal")


# Anomaly Scores over time

st.subheader('Anomaly Scores')
fig_scores= px.line(df_live , x='timestamp', y='scores', color='user_id', title='Anomaly Scores By Users Over Time')
st.plotly_chart(fig_scores, use_container_width=True)


# Alerts Panel

st.subheader("Alerts Panel")


alerts_df = df_live[df_live["alerts"] == True]

if len(alerts_df) == 0:
    st.success("No anomalies detected.")
else:
    st.error(f"{len(alerts_df)} anomalies detected!")

    # Show only important columns
    display_cols = [
        "user_id",
        "amount",
        "scores"
    ]

    # Sort by highest anomaly score
    alerts_df = alerts_df.sort_values(by="scores", ascending=False)

    st.dataframe(alerts_df[display_cols], use_container_width=True)


# Rolling Stats

st.subheader('Rolling Statistics')
fig_rolling= px.line(df_live, x='timestamp', y=['rolling_mean_5min', 'user_rolling_mean_10min'], title='Global & Per-User Rolling Means')
st.plotly_chart(fig_rolling, use_container_width=True)


#Per-User Risk Summary Table

summary = df_live.groupby("user_id").agg(
    total_transactions=("amount", "count"),
    total_alerts=("alerts", "sum"),
    avg_amount=("amount", "mean"),
    max_score=("scores", "max")
).reset_index()

st.subheader("Per-User Risk Summary")
st.dataframe(summary)


# Concept Drift Points

st.subheader('Concept Drift Detected')
for drift_time in drift_points:
    st.markdown(f'Concept Drift Detected at {drift_time}')



