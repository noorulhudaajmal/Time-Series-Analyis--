import keras
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
import streamlit as st

#--------------------------------------------------------------------------------------
temp = """
<div style="background-color:{};padding:0.5px;border-radius:5px;">
<h4 style="color:{};text-align:center;display:in-line">{}</h6>
</div>
"""

def get_chart(data):
    fig = go.Figure(
        data = [go.Table ( columnwidth=[3,4,4,4,3],
                           header = dict(values = list(data.columns),font=dict(size=12, color = 'white'),fill_color = '#264653',line_color = 'rgba(255,255,255,0.2)',align = ['left','center'],height=40)
                           ,cells = dict(values = [data[K].tolist() for K in data.columns],font=dict(size=12 , color = "black"),align = ['left','center'],
                                         line_color = 'rgba(255,255,255,0.2)',height=30))])
    fig.update_layout(title_font_color = '#264653',title_x=0,margin= dict(l=0,r=10,b=10,t=30), height=480 , width=600)
    return fig




#---------------------------------------------------------------------------------------

st.set_page_config("Time-Series-Analysis",layout="wide")

# --------------------------------------------------------------------------------------

data = pd.read_csv("data/data_.csv")
data_ = pd.read_csv("data/data.csv")
# st.dataframe(data)

colors = ["#354f52","#52796f","#84a98c","#cad2c5"]
#---------------------------------------------------------------------------------------

st.markdown(temp.format('#66999B','white' , "TIME SERIES ANALYSIS"),unsafe_allow_html=True)

st.write("#### DATA OVERVIEW")
actual,updated = st.columns((1,1))
with actual:
    st.write("Actual observed data...")
    chart = get_chart(data_)
    st.plotly_chart(chart)
with updated:
    st.write("Data updated by converting time to have 5 sec interval...")
    chart = get_chart(data)
    st.plotly_chart(chart)

#---------------------------------------------------------------------------------------
st.markdown(temp.format('#66999B','white' , "An overview of observations along the time axis..."),unsafe_allow_html=True)
fig = px.line(y = data.obs , x = data.Time,
              labels={"x":"Time","y":"Reading"},
              color_discrete_sequence=colors,
              width=1200)
fig.update_layout(xaxis = dict(rangeslider=dict(visible=True)))
st.plotly_chart(fig)
#---------------------------------------------------------------------------------------
st.markdown(temp.format('#66999B','white' , "PLAN OF ATTACK"),unsafe_allow_html=True)
st.write()
st.markdown(
    """
    Analyze the time series data in the following steps:
    - Standardizing the Observations, so that there is one unit variance; This will be done using sklean.StandardScalar()
    - Make the dataset ready to feed the Time-Series-ML-model. For this purpose it needs to be in 3D. [Number of rows,Number of steps,NUmber of columns]
    - Split the dataset into 70-30 ratio, for training and testing the ML model.
    - Training the LSTM model
    - Get the model prediction over the whole Observations.
    - Comparing prediction/reproduced values with actual values.
    - Setting the error maximum threshold (Deviation of actual from prediction) to be 30%.Any observation with deviation > 0.30 will be considered as Anomalous.
    - Visualizing the results by plotting actual and reproduced curves with anomaly points highlighted.
    - Conclusion 
    """
)
#---------------------------------------------------------------------------------------
st.markdown(temp.format('#66999B','white' , """1. STANDARDIZING"""),unsafe_allow_html=True)
st.write()
X = data["obs"].values
X = X.reshape(-1,1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

desc,vals = st.columns((1,1))
with desc:
    st.write("###### \n\n\n This is something the data values look like...>")
with vals:
    temp_frame = pd.DataFrame();temp_frame["Time"] = data.Time;temp_frame["Obs"] = X;
    st.plotly_chart(get_chart(temp_frame))

#---------------------------------------------------------------------------------------
st.markdown(temp.format('#66999B','white' , """2. 2D DATA TO 3D"""),unsafe_allow_html=True)
st.write()
def preprocess(X, time_steps=5):
    x_values = []
    y_values = []

    for i in range(len(X)-time_steps):
        x_values.append(X[i:(i+time_steps)])
        y_values.append(X[(i+time_steps)])

    return np.array(x_values), np.array(y_values)
X,y = preprocess(X)
st.write("""
The transformation can be understood as;
- Take an appropriate step size-I chosed it to be 5.
- Split the whole data in a way that input array having value at index 0 is first 5 obs values, index 1 has 2-6 obs values, index 3 has 3-7 obs values from origional observations.
- input[0] = obs[0:4] , output[0] = obs[5] ; input[1] = obs[1:5], output[1] = obs[6] and so on....
""")
inp , out = st.columns((1,1))

with inp:
    st.write("#### Input array")
    st.write(X[:3])
with out:
    st.write("#### Output array")
    st.write(y)


#---------------------------------------------------------------------------------------
st.markdown(temp.format('#66999B','white' , """3. 70:30 TRAIN-TEST SPLIT"""),unsafe_allow_html=True)
st.write()
X_train , X_test , y_train, y_test = train_test_split(X,y,train_size=0.70,shuffle=False)
st.write("""
The shapes of input-output arrays are as follows:
- Training input : {0}
- Training output : {1}
- Testing input : {2}
- Testing output : {3}
""".format(X_train.shape , y_train.shape,X_test.shape,y_test.shape))
# hist = px.histogram(data,X.flatten())
# st.plotly_chart(hist)

#---------------------------------------------------------------------------------------
st.markdown(temp.format('#66999B','white' , """4. Training the LSTM model"""),unsafe_allow_html=True)
st.write()
mdl = st.selectbox(options = ["Model-2","Model-1"],label="Select model")
st.write("Model-1 has higher regularization factor than Model-2, but both gives having MAE ~= 1")
st.image("MAEs.png")
if mdl=="Model-1":
    model = keras.models.load_model("model/model.h5")
else:
    model = keras.models.load_model("model/model1.h5")
st.write("The summary of trained LSTM model is as follows :")
summ,curve = st.columns((1,1))
with summ:
    st.write("##### MODEL SUMMARY")
    st.image("model1_summary.png")
with curve:
    st.write("##### LOSS CURVES")
    st.write("*** of Model-1")
    st.image("loss.png")

#---------------------------------------------------------------------------------------
st.markdown(temp.format('#66999B','white' , """5. Get Expected observations"""),unsafe_allow_html=True)
st.write()
st.write("This is done by passing the observation values to .predict() function of the model. It gives the expected behaviour of the system.")
expected = model.predict(X)

comp_df=pd.DataFrame(); comp_df["Time"]=data.Time[5:]; comp_df["Actual"]=y; comp_df["Expected"]=expected;
st.plotly_chart(get_chart(comp_df))

#---------------------------------------------------------------------------------------
st.markdown(temp.format('#66999B','white' , """5. Comparison of Actual and Reproduced/Expected"""),unsafe_allow_html=True)
st.write()
fig = go.Figure(data=go.Scatter(x=comp_df.Time,y=comp_df.Actual,name='Actual',
                                line = dict(color=colors[2])))
fig.add_trace(go.Scatter(x=comp_df.Time,y=comp_df.Expected,name='Expected',
                         line = dict(color=colors[0], dash='dash')))
# fig.add_trace(px.scatter(x=anamolous.Time,y=frame.obs))
# fig = px.line(x=frame.Time,y=frame.obs)
fig.update_layout(xaxis = dict(rangeslider=dict(visible=True)),width=1200,
                  hoverlabel = dict(
                      bgcolor="white",
                      font_size=16,
                      font_family="Rockwell"
                  ),hovermode="x unified")

st.plotly_chart(fig)

#---------------------------------------------------------------------------------------
st.markdown(temp.format('#66999B','white' , """5. Getting deviations and finding Anomalous observations"""),unsafe_allow_html=True)
st.write()
MAE = np.mean(np.abs(y - expected), axis=1)
comp_df["Divergence"] = MAE
comp_df["Anomaly"] = comp_df.Divergence>0.3
st.plotly_chart(get_chart(comp_df))

#---------------------------------------------------------------------------------------
st.markdown(temp.format('#66999B','white' , """5. VISUALIZING THE ANOMALY"""),unsafe_allow_html=True)
st.write()

anamolous = comp_df[comp_df.Anomaly==True]
fig = go.Figure(data=go.Scatter(x=comp_df.Time,y=comp_df.Actual,name='Actual',
                                line = dict(color=colors[2])))
fig.add_trace(go.Scatter(x=anamolous.Time, y=anamolous.Actual,name="Anomaly", mode="markers",
                         marker=dict(color="orange",size=3)))
fig.update_layout(xaxis = dict(rangeslider=dict(visible=True)),width=1200,
                  hoverlabel = dict(
                      bgcolor="white",
                      font_size=16,
                      font_family="Rockwell"
                  ),hovermode="x unified")
st.plotly_chart(fig)
#---------------------------------------------------------------------------------------
st.markdown(temp.format('#66999B','white' , """5. CONCLUSION"""),unsafe_allow_html=True)
st.write()
st.write("""The deviation threshold is set to be accordance to the number of outliers in the data.
So this interpretation tells that according to the series of data the expected behaviour at some instance
is different from the observed one, which is marked as anomalous.""")
