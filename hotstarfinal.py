import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("notebook")

plt.rcParams['figure.titlesize'] = 24
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# ── Load & Clean ──────────────────────────────────────────────
df = pd.read_csv(r"C:\Users\HP\Downloads\data_analytics\disney_shows.csv")

df['director']    = df['director'].fillna('Unknown')
df['country']     = df['country'].fillna('Unknown')
df['actors']      = df['actors'].fillna('Unknown')
df['rated']       = df['rated'].fillna('Not Rated')
df['imdb_rating'] = pd.to_numeric(df['imdb_rating'], errors='coerce')

df.drop_duplicates(inplace=True)
df_clean = df.dropna(subset=['type', 'year'])

# Extract start year from ranges like "2018–"
df_clean = df_clean.copy()
df_clean['start_year'] = df_clean['year'].astype(str).str.extract(r'(\d{4})').astype(float)

avg_rating = df_clean['imdb_rating'].mean()

total_records  = len(df_clean)
total_movies   = len(df_clean[df_clean['type'] == 'movie'])
total_series   = len(df_clean[df_clean['type'] == 'series'])

print("Total Records  :", total_records)
print("Movies         :", total_movies)
print("Series         :", total_series)
print("Avg IMDB Rating:", round(avg_rating, 2))

# ── Figure Setup ──────────────────────────────────────────────
fig, axes = plt.subplots(
    3, 3,
    figsize=(24, 18),
    constrained_layout=True,
    facecolor='white'
)

fig.suptitle(
    "DISNEY+ DATA ANALYSIS DASHBOARD",
    fontsize=24,
    fontweight='bold'
)

for row in axes:
    for ax in row:
        ax.set_facecolor('#F8F9FA')
        ax.grid(alpha=0.3)

# ── Plot 1 : Movies vs Series (Bar) ───────────────────────────
sns.countplot(
    data=df_clean,
    x='type',
    ax=axes[0, 0],
    palette='Blues_d'
)
axes[0, 0].set_title("Movies vs Series")
axes[0, 0].set_xlabel("Content Type")
axes[0, 0].set_ylabel("Count")

# ── Plot 2 : Top 10 Countries (Horizontal Bar) ────────────────
top_country = df_clean['country'].value_counts().head(10)
sns.barplot(
    x=top_country.values,
    y=top_country.index,
    ax=axes[0, 1],
    palette='Blues_d'
)
axes[0, 1].set_title("Top 10 Countries")
axes[0, 1].set_xlabel("Count")
axes[0, 1].set_ylabel("Country")

# ── Plot 3 : Top Ratings (Horizontal Bar) ─────────────────────
sns.countplot(
    data=df_clean,
    y='rated',
    order=df_clean['rated'].value_counts().index[:10],
    ax=axes[0, 2],
    palette='Blues_d'
)
axes[0, 2].set_title("Top Content Ratings")
axes[0, 2].set_xlabel("Count")
axes[0, 2].set_ylabel("Rating")

# ── Plot 4 : Content Growth by Year (Line) ────────────────────
year_data = df_clean['start_year'].dropna().value_counts().sort_index()
axes[1, 0].plot(
    year_data.index,
    year_data.values,
    marker='o',
    linewidth=2,
    color='steelblue'
)
axes[1, 0].set_title("Content Growth by Year")
axes[1, 0].set_xlabel("Release Year")
axes[1, 0].set_ylabel("Count")

# ── Plot 5 : Top Genres (Horizontal Bar) ──────────────────────
genres = df_clean['genre'].dropna().str.split(',').explode().str.strip()
top_genres = genres.value_counts().head(10)
sns.barplot(
    x=top_genres.values,
    y=top_genres.index,
    ax=axes[1, 1],
    palette='Blues_d'
)
axes[1, 1].set_title("Top Genres")
axes[1, 1].set_xlabel("Count")
axes[1, 1].set_ylabel("Genre")

# ── Plot 6 : Content Percentage (Pie) ─────────────────────────
type_counts = df_clean['type'].value_counts()
axes[1, 2].pie(
    type_counts.values,
    labels=type_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.75,
    labeldistance=1.15,
    textprops={'fontsize': 10}
)
axes[1, 2].set_title("Content Percentage")

# ── Plot 7 : Top Release Years (Bar) ──────────────────────────
top_years = df_clean['start_year'].dropna().value_counts().head(10)
sns.barplot(
    x=top_years.index.astype(int).astype(str),
    y=top_years.values,
    ax=axes[2, 0],
    palette='Blues_d'
)
axes[2, 0].set_title("Top Release Years")
axes[2, 0].set_xlabel("Year")
axes[2, 0].set_ylabel("Count")
axes[2, 0].tick_params(axis='x', rotation=45)

# ── Plot 8 : Top 5 Countries Share (Pie) ──────────────────────
country_data = df_clean['country'].value_counts().head(5)
axes[2, 1].pie(
    country_data.values,
    labels=country_data.index,
    autopct='%1.1f%%',
    startangle=90,
    pctdistance=0.65,
    labeldistance=1.35,
    textprops={'fontsize': 8},
    wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}
)
axes[2, 1].set_title("Top 5 Countries Share")
axes[2, 1].patch.set_edgecolor('black')
axes[2, 1].patch.set_linewidth(2)
for spine in axes[2, 1].spines.values():
    spine.set_visible(True)

# ── Plot 9 : Summary Box ──────────────────────────────────────
axes[2, 2].axis('off')

summary_text = (
    f"Total Records      : {total_records}\n"
    f"Total Movies       : {total_movies}\n"
    f"Total Series       : {total_series}\n"
    f"Avg IMDB Rating    : {round(avg_rating, 2)}\n"
    f"Common Rating      : {df_clean['rated'].mode()[0]}"
)

axes[2, 2].text(
    0.5, 0.5,
    summary_text,
    ha='center',
    va='center',
    fontsize=12,
    fontweight='bold',
    linespacing=1.8,
    bbox=dict(
        boxstyle='round,pad=1.2',
        facecolor='#D6EAF8',
        edgecolor='#3498DB',
        linewidth=2
    )
)
axes[2, 2].set_title("Project Summary", pad=20)

plt.show()
