import os
from pathlib import Path
from urllib import response
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import OpenAI
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import streamlit as st
import pandas as pd


# Load environment variables
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
DATA_PATH = Path(__file__).resolve().parent / "complaints_train.csv"
KNOWLEDGE_BASE_PATH = Path(__file__).resolve().parent / "knowledge_base.json"

import matplotlib.pyplot as plt


st.set_page_config(page_title="Complaint Dashboard", layout="wide")

df_complaints = pd.read_csv(DATA_PATH)

df_complaints['timestamp_update'] = pd.to_datetime(df_complaints['timestamp'], format='mixed')
st.title("Complaint Dashboard")

with open("model_classification_report.txt", "r") as f:
    st.code(f.read())

category_counts = df_complaints['category'].value_counts()
fig, ax = plt.subplots(figsize=(15, 5))
ax.bar(category_counts.index, category_counts.values)
ax.set_title("Distribution of Categories")
ax.set_xlabel("Category Type")
ax.set_ylabel("Count")
ax.tick_params(axis='x', rotation=90)
fig.tight_layout()
st.pyplot(fig, use_container_width=False)
plt.close(fig)

st.subheader("Analysis:")
st.write("Above we you can see the distribution of category types. The data shows that the majority of complaints fall" \
" under the Delivery Delay category, while the least common is Documentation error\n\n.")

df_category_sorted= df_complaints.sort_values(by=['region', 'category'])
fig, ax = plt.subplots(figsize=(15, 5))
sns.countplot(data=df_category_sorted, x='region', hue='category', ax=ax)
ax.set_title("Complaints by Region and Category")
ax.set_xlabel("Region")
ax.set_ylabel("Count")
ax.legend(title="Category", loc='center left', bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.subheader("Analysis:")
st.write("The above chart shows the distribution of complaints by region and category. Each region has a similar number of complaints." \
         " The most common complaints are delivery delays and damaged goods.\n\n")
st.markdown("- Asia: This region has the least quality issue and service complaints")
st.markdown("- Europe: This region has the least poor communication complaints but it has a good amount of missing item complaints")
st.markdown("- North America: This region has the least documentation error complaints but it has the most service complaints")
st.write("\n\n")

df_category_sorted= df_complaints.sort_values(by=['channel', 'category'])
fig, ax = plt.subplots(figsize=(15, 5))
sns.countplot(data=df_category_sorted, x='channel', hue='category', ax=ax)
ax.set_title("Complaints by Channel and Category")
ax.set_xlabel("Channel")
ax.set_ylabel("Count")
ax.legend(title="Category", loc='center left', bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.subheader("Analysis:")
st.write("The above chart shows the distribution of complaints by channel and category. Each channel has a similar number of complaints." \
         " The most common complaints are delivery delays.\n\n")
st.markdown("- App: This channel has a low nu,mber of documentation errors and quality issues. Compared to the other channels it has a lot of pricing error compliants.")
st.markdown("- Email: This channel has the least poor communication complaints but it has a the most missing item complaints")
st.markdown("- Social: This channel has the least documentation error complaints but it has the most damaged goods and quality issues complaints")
st.markdown("- Web: This channel has a low ammount of documentation errors and poor communication complaints")
st.write("\n\n")

category_counts = df_complaints['channel'].value_counts()
fig, ax = plt.subplots(figsize=(15, 5))
ax.bar(category_counts.index, category_counts.values)
ax.set_title("Distribution of Channels")
ax.set_xlabel("Channel Type")
ax.set_ylabel("Count")
ax.tick_params(axis='x', rotation=90)
fig.tight_layout()
st.pyplot(fig, use_container_width=False)
plt.close(fig)

st.subheader("Analysis:")
st.write("Above we you can see the distribution of channel types. The data shows that the majority of complaints fall" \
" under the social channel, while the least common is the Web channel. But overall each channel has a similar number of complaints.\n\n.")

df_complaints['hour'] = df_complaints['timestamp_update'].dt.hour
df_complaints_filtered_morning = df_complaints[df_complaints['hour'] < 13].copy()
df_category_sorted= df_complaints_filtered_morning.sort_values(by=['hour', 'category'])
fig, ax = plt.subplots(figsize=(15, 3))
sns.countplot(data=df_category_sorted, x='hour', hue='category', ax=ax)
ax.set_title("Complaints by Hour and Category")
ax.set_xlabel("Hour")
ax.set_ylabel("Count")
ax.legend(title="Category", loc='center left', bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

df_complaints_filtered_afternoon = df_complaints[df_complaints['hour'] > 12].copy()
df_category_sorted= df_complaints_filtered_afternoon.sort_values(by=['hour', 'category'])
fig, ax = plt.subplots(figsize=(15, 3))
sns.countplot(data=df_category_sorted, x='hour', hue='category', ax=ax)
ax.set_title("Complaints by Hour and Category")
ax.set_xlabel("Hour")
ax.set_ylabel("Count")
ax.legend(title="Category", loc='center left', bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

st.subheader("Analysis:")
st.markdown("- Damaged Goods: The highest number occurs at midnight and the least amount occured at 6/7 am. " \
"At 6/7 am delivery companies are probably not likely to try to drop off a package because poeple not awake or on their way to work. So people woudln't have access to the package yet to determine if it's damaged.")
st.markdown("- Delivery Delays: The highest number occurs at 3 am and 5 pm. The 5pm time makes sense beacuase people are coming home from work and have expected a pakage to have arrived during the day.")
st.markdown("- Documentation Error: These are pretty low overall, the lowest being at 1 am and the highest being at 9/10 pm. At 1 am most people are not awake to make document an error occuring. At 9/10 pm you might be user that hase being working on a product for a while and you might finally have to submkit a complaint.")
st.markdown("- Missing Items: The lowest number occurs between 3pm and 8pm. The highest occurs at 2pm. If you are getting a package in the middle of the day you might find something is missing from it and submit a complaint after that. ")
st.markdown("- Poor Communication: These number of kind of up and down across the day. They get lower towards lunch time.")
st.markdown("- Pricing Error: The least amount is at 8pm and the most occurs at 6am. If it's 6 am and you are starting your day you might see and e-mail or alert saying something about a price issue from you bank. If it's 8pm most people are busying with dinner, family time, etc and don't have the time go through pricing issues to make a complaint. ")
st.markdown("- Service Complaint: The lowest number occurs between 12pm and 2pm.The most occur at 11am.  ")
st.markdown("- Quality Issue: The lowest number occurs between 4pm. The highest occurs at 3pm. ")


