# Football Scouting Engine: Player Similarity and Role-Based Evaluation

A data-driven scouting tool that finds players who play like a target player, and evaluates players against their positional role instead of raw totals.

## Core question
Given a player a club wants (or is about to lose), which other players play most similarly, and who performs best for their role relative to peers?

## Scope (v1, keep it tight)
- One data source, one or two competitions, one clear deliverable.
- Outfield players only (goalkeepers need a separate metric set).
- Minimum minutes filter (e.g. 900) so small samples don't dominate.

## Data
- Primary: StatsBomb Open Data (free, event-level, on GitHub) via `statsbombpy`.
- Limitation: open data covers selected competitions and seasons, not every league. Check what is available for the competition you pick before committing.
- Optional later: add a second source for market value or wages to enable a "value" angle.

## Pipeline
1. Ingest: pull events and lineups for chosen competition(s), cache to `data/raw/`.
2. Clean: aggregate events to player-season level, compute per-90 metrics, filter by minutes.
3. Features: progressive passes, shot-creating actions, pressures, tackles+interceptions, carries, xG and xA, aerial duels, etc., grouped by role.
4. Roles: cluster players (e.g. K-Means or Gaussian Mixture on PCA-reduced features) into data-driven roles; validate clusters against listed positions.
5. Similarity: cosine or Euclidean distance within standardized feature space; return top-N similar players with a breakdown of why.
6. Evaluation: percentile ranks within role; composite score per role with transparent weights.
7. App: Streamlit dashboard with player search, radar chart, similar-players table, role leaderboard.

## Repo structure
```
football-scouting/
  data/raw/  data/processed/
  notebooks/   01_explore.ipynb  02_features.ipynb  03_roles.ipynb  04_similarity.ipynb
  src/         ingest.py  features.py  roles.py  similarity.py
  app/         streamlit_app.py
  README.md  requirements.txt
```

## Milestones
1. Week 1: environment, pull data, explore what exists, first per-90 table.
2. Week 2: feature engineering and role clustering.
3. Week 3: similarity search and percentile scoring, with sanity checks (does it return sensible players for known names?).
4. Week 4: Streamlit app, write-up, README with screenshots, publish to GitHub.

## What will make this stand out on a CV
- A clear written methodology and honest limitations section.
- Validation: show the model recovers known similar pairs, not just pretty charts.
- A live, interactive dashboard and a clean repo.

## Requirements (starting point)
pandas, numpy, scikit-learn, statsbombpy, matplotlib, mplsoccer, streamlit, plotly
