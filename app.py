import streamlit as st
import st_pages
from st_pages import show_pages_from_config, add_page_title,Page, Section, show_pages
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from PIL import Image


rcParams['font.family'] = 'Microsoft YaHei'

add_page_title()

show_pages(
    [
        Page("crawl-data.py", "Data Crawling", ":books:"),
        Page("app.py", "Home", "🏠"),
        Page("PCA-factor.py", "PCA and Factor Analysis", ":books:"),
        Page("boosting-bagging.py", "Machine Learning Model Prediction", ":books:"),
    ]
)

st.title('ESG rating')
st.markdown('彭宁一 12012244  程民欣 12012238  赵远帆 12012809')

df1 = pd.read_csv(".\\input\\shares.csv")
df2 = pd.read_excel(".\\input\\ESG.xlsx")
df = df1.join(df2, on = 'id',how = 'inner',rsuffix = '_hh')

df.set_index('id', inplace = True)

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    modify = st.checkbox("Add filters")

    if not modify:
        return df
    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

st.dataframe(filter_dataframe(df))

st.markdown('''We plotted a boxplot of the ESG scores for each dimension to see the distribution of
scores.
''')

df1 = pd.read_csv(".\\input\\shares.csv")
df2 = pd.read_excel(".\\input\\ESG.xlsx")
df = df1.join(df2, on = 'id',how = 'inner',rsuffix = '_hh')

df3 = df[['ESG管理实践得分\n[交易日期] 2023-05-11', 'ESG争议事件得分\n[交易日期] 2023-05-11', \
       '环境维度得分\n[交易日期] 2023-05-11', '社会维度得分\n[交易日期] 2023-05-11', \
       '治理维度得分\n[交易日期] 2023-05-11']].melt(var_name="quartilemethod")  # 宽表转成长表

fig2 = plt.figure(figsize=(10, 10))
sns.boxplot(data=df3, x="value", y="quartilemethod")
st.pyplot(fig2)


st.markdown('''We performed a correlation analysis of the underlying stock financial data using pariplot
and hue is the ESG Rating.
''')

df.rename(columns={'ESG管理实践得分\n[交易日期] 2023-05-11':'管理实践得分', 'ESG争议事件得分\n[交易日期] 2023-05-11':'争议事件得分',
       '环境维度得分\n[交易日期] 2023-05-11':'环境维度得分', '社会维度得分\n[交易日期] 2023-05-11':'社会维度得分',
       '治理维度得分\n[交易日期] 2023-05-11':'治理维度得分'}, inplace=True)
sub_rating = ['管理实践得分', '争议事件得分','环境维度得分', '社会维度得分','治理维度得分']



# fig3 = plt.figure(figsize=(10, 10))
# sns.pairplot(df[['最新价', '涨跌幅(%)','市现率', 'ROE', '负债率(%)','PE(动)', '净利润',"Wind ESG评级\n[交易日期] 2023-05-11"]], hue = "Wind ESG评级\n[交易日期] 2023-05-11")
# st.pyplot(fig3)

image = Image.open('.//input//pairplot.png')
st.image(image)


def zscore_normalize(data):
    mu     = np.mean(data, axis=0)                 # mu will have shape (n,)
    sigma  = np.std(data, axis=0)                  # sigma will have shape (n,)
    X_norm = (data - mu) / sigma
    return X_norm


stock=pd.read_excel('.\\input\\股票数据.xlsx')
esg_wind_sz=pd.read_excel(r'.\\input\\深股1.xlsx')
esg_wind_sh=pd.read_excel(r'.\\input\\上股1.xlsx')

#合并深股上股
esg_wind=pd.concat([esg_wind_sz,esg_wind_sh],axis=0)
esg_wind=esg_wind.reset_index().drop('index',axis=1)
#去除esg数据中的异常值
esg_wind=esg_wind.dropna()
#合并股票数据和ESG数据
df=pd.merge(stock, esg_wind, left_on='股票简称',right_on='证券简称', how='inner')

df=df.drop('证券简称',axis=1)
df1=df.drop(['股票代码','证券代码','股票简称','商道融绿ESG评级','华证ESG评级','所属行业','Wind ESG评级','Wind ESG综合得分'],axis=1)

#对数值型变量标准化
df_normal=zscore_normalize(df1)

df2=df[['Wind ESG综合得分','最新价', '涨跌幅(%)', 'PE(TTM)', 'PE(静)', '市净率', 'PEG值',
       '市销率', '市现率','收益', '总股本', 'ROE(%)','PE(动)', '净利润', '总营收', 'ESG管理实践得分', 'ESG争议事件得分', '环境维度得分',
       '社会维度得分', '治理维度得分']]


st.markdown('''
Heatmaps can show correlations between different features combining the ESG data and
the stock basic data.
''')
ve_corr = df2.corr()
ve_corr
mask = np.zeros_like(ve_corr)
mask[np.triu_indices_from(mask)] = True
fig = plt.figure(figsize=(10, 10))
sns.heatmap(ve_corr, cmap='Spectral', vmax=1.0, vmin=-1.0 , mask = mask, linewidths=2)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
st.pyplot(fig)

# violin plot
# fig0 = plt.figure(figsize=(10, 10))
# sns.violinplot(x=df['Wind ESG综合得分'])
# st.pyplot(fig0)
# box plot

st.markdown('''
We can see the correspondence of the
indicators between overall scores and ratings through a box plot''')
df_order=df.sort_values('Wind ESG评级')
fig1 = plt.figure(figsize=(10, 10))
sns.boxplot(x=df_order['Wind ESG评级'], y=df_order['Wind ESG综合得分'])
st.pyplot(fig1)
