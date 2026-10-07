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

df_complaints['hour'] = df_complaints['timestamp_update'].dt.hour
df_category_sorted= df_complaints.sort_values(by=['hour', 'category'])
fig, ax = plt.subplots(figsize=(15, 5))
sns.countplot(data=df_category_sorted, x='hour', hue='category', ax=ax)
ax.set_title("Complaints by Hour and Category")
ax.set_xlabel("Hour")
ax.set_ylabel("Count")
ax.legend(title="Category", loc='center left', bbox_to_anchor=(1.0, 0.5))
fig.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close(fig)