"""Pull EPL 2015/16 from StatsBomb open data and cache it locally.

Usage: python src/ingest.py
Writes to data/raw/: matches.pkl, events.pkl, lineups.pkl
"""
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from statsbombpy import sb

warnings.filterwarnings("ignore")

COMPETITION_ID = 2   # Premier League
SEASON_ID = 27       # 2015/16
RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


KEEP = [
    "match_id", "index", "period", "minute", "second", "type", "team",
    "player_id", "player", "position", "under_pressure", "counterpress",
    "location", "pass_end_location", "carry_end_location", "pass_length",
    "pass_outcome", "pass_shot_assist", "pass_goal_assist", "pass_cross",
    "pass_through_ball", "pass_recipient_id", "pass_aerial_won",
    "shot_outcome", "shot_statsbomb_xg", "shot_type", "shot_aerial_won",
    "dribble_outcome", "duel_type", "duel_outcome", "clearance_aerial_won",
    "miscontrol_aerial_won", "interception_outcome", "foul_committed_card",
    "ball_recovery_recovery_failure",
]
LOC_COLS = ["location", "pass_end_location", "carry_end_location"]


def slim(events: pd.DataFrame) -> pd.DataFrame:
    events = events[[c for c in KEEP if c in events.columns]].copy()
    for c in LOC_COLS:
        if c in events.columns:
            xy = pd.DataFrame(
                events[c].apply(lambda v: v if isinstance(v, list) and len(v) >= 2 else [None, None]).tolist(),
                index=events.index,
            ).iloc[:, :2]
            events[c + "_x"] = pd.to_numeric(xy[0], errors="coerce").astype("float32")
            events[c + "_y"] = pd.to_numeric(xy[1], errors="coerce").astype("float32")
            events = events.drop(columns=c)
    for c in events.columns:
        if events[c].dtype == object and c not in ("type", "team", "player", "position"):
            events[c] = events[c].astype("category") if events[c].nunique() < 50 else events[c]
    return events


def fetch_match(match_id: int):
    events = sb.events(match_id=match_id)
    events["match_id"] = match_id
    events = slim(events)
    lineups = sb.lineups(match_id=match_id)
    rows = []
    for team, df in lineups.items():
        df = df.copy()
        df["team"] = team
        df["match_id"] = match_id
        rows.append(df)
    return events, pd.concat(rows, ignore_index=True)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    matches.to_pickle(RAW / "matches.pkl")
    ids = matches["match_id"].tolist()
    print(f"{len(ids)} matches")

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(fetch_match, ids))

    events = pd.concat([r[0] for r in results], ignore_index=True)
    lineups = pd.concat([r[1] for r in results], ignore_index=True)
    events.to_pickle(RAW / "events.pkl")
    lineups.to_pickle(RAW / "lineups.pkl")
    print(f"events: {len(events):,} rows, lineups: {len(lineups):,} rows")


if __name__ == "__main__":
    main()
