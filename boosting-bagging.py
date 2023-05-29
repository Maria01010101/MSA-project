import streamlit as st
import st_pages
from st_pages import show_pages_from_config, add_page_title,Page, Section, show_pages
import numpy as np
import pandas as pd
from PIL import Image

st.header('Score Prediction')

st.markdown('''
The three ensemble models' results are shown as below. The feature importance is a list and in the order of [管理实践得分,治理维度得分,争议事件得分,社会维度得分,环境维度得分]
''')

st.latex(r'''
\begin{equation}
\begin{array}{|c|c|c|c|}
\hline \text { Method } & \text { R-squared } & \text { MSE } & \text { Feature importance } \\
\hline \text { Random Forest } & 0.69 & 0.22 & {[0.79,0.0640 .058,0.051,0.041]} \\
\hline \text { AdaBoost } & 0.68 & 0.23 & {[0.909,8.038,0.011,0.023,8.019]} \\
\hline \text { Xgboost } & 0.67 & 0.23 & {[0.716,0.874,0.0685,0.872,0.869]} \\
\hline
\end{array}
\end{equation}
''')


st.markdown('''The neural network model we trained uses two hidden layers with 32 hidden units and
one output layer. Parameters: learning rate is 0.001, alpha is 0.01.
The R-squared of Neural Network Model can near to 0.99. And the loss of each
epoches are shown as below:
23
''')

image = Image.open('.//input//loss.png')
st.image(image, caption='loss')

st.markdown(''' 

The classification report for the four methods:
''')


st.latex(r'''
\begin{equation}
\begin{array}{|c|c|c|c|c|}
\hline \text { Method } & \text { Precision } & \text { Recall } & \text { F1-score } & \text { Accuracy } \\
\hline \text { Support Vector Machine } & 0.74 & 0.70 & 0.71 & 0.70 \\
\hline \text { Muti-Logistic Regression } & 0.74 & 0.70 & 0.71 & 0.70 \\
\hline \text { Extra Tree Classifier } & 0.77 & 0.76 & 0.76 & 0.76 \\
\hline \text { Self-organizing map } & 0.69 & 0.70 & 0.69 & 0.70 \\
\hline
\end{array}
\end{equation
''')


st.markdown(''' 
The clustering results obtained using the SOM method are shown in the figure below. The output grid size is calculated by the empirical formula $\sqrt{5\times \sqrt{N}}$, and N is the sample size
''')

image = Image.open('.//input//som classification.png')
st.image(image, caption='SOM')




