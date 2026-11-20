"""Build a player-season table (minutes, counts, per-90 metrics) from cached raw data.

Usage: python src/features.py
Writes to data/processed/: player_season.csv (all players), player_per90.csv (outfield, >= MIN_MINUTES)
Pitch coordinates are StatsBomb's 120 x 80, attacking left to right.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
MIN_MINUTES = 900

WON = {"Won", "Success In Play", "Success Out"}


def clock_to_min(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    m, sec = str(s).split(":")[:2]
    return int(m) + int(sec) / 60


def position_group(pos: str) -> str:
    p = pos.lower()
    if "goalkeeper" in p:
        return "GK"
    if "center back" in p:
        return "CB"
    if "wing back" in p or p.endswith(" back"):
        return "FB"
    if "defensive midfield" in p:
        return "DM"
    if "attacking midfield" in p:
        return "AM"
    if "wing" in p or "left midfield" in p or "right midfield" in p:
        return "W"
    if "midfield" in p:
        return "CM"
    if "forward" in p or "striker" in p:
        return "FW"
    return "OTHER"


def minutes_table(lineups: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Minutes per player per match from lineup position spells; None end = final whistle."""
    end = (
        events.assign(t=events["minute"] + events["second"] / 60)
        .groupby("match_id")["t"].max().rename("match_end")
    )
    rows = []
    for r in lineups.itertuples(index=False):
        for p in r.positions:
            start = clock_to_min(p["from"])
            stop = clock_to_min(p["to"])
            rows.append((r.match_id, r.player_id, r.player_name, r.team, p["position"], start, stop))
    sp = pd.DataFrame(rows, columns=["match_id", "player_id", "name", "team", "position", "start", "stop"])
    sp = sp.merge(end, left_on="match_id", right_index=True)
    sp["stop"] = sp["stop"].fillna(sp["match_end"])
    sp["minutes"] = (sp["stop"] - sp["start"]).clip(lower=0)
    return sp


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    e = pd.read_pickle(RAW / "events.pkl")
    lineups = pd.read_pickle(RAW / "lineups.pkl")
    e = e[e["player_id"].notna()].copy()
    e["player_id"] = e["player_id"].astype(int)

    spells = minutes_table(lineups, e)
    minutes = spells.groupby("player_id")["minutes"].sum().rename("minutes")
    info = (
        spells.groupby(["player_id", "name", "team"])["minutes"].sum().reset_index()
        .sort_values("minutes", ascending=False).drop_duplicates("player_id")
        .set_index("player_id")[["name", "team"]]
    )
    pos_min = spells.groupby(["player_id", "position"])["minutes"].sum().reset_index()
    pos_min["group"] = pos_min["position"].map(position_group)
    main_pos = (
        pos_min.sort_values("minutes", ascending=False).drop_duplicates("player_id")
        .set_index("player_id")[["position", "group"]]
        .rename(columns={"position": "main_position", "group": "pos_group"})
    )

    x, y = e["location_x"], e["location_y"]
    in_box = (x >= 102) & y.between(18, 62)
    final_third = x >= 80
    typ = e["type"]

    def cnt(mask):
        return e.loc[mask].groupby("player_id").size()

    # Passes
    is_pass = typ == "Pass"
    pass_ok = is_pass & e["pass_outcome"].isna()
    gain = e["pass_end_location_x"] - x
    prog_pass = pass_ok & (gain >= 10) & (x >= 48)
    box_pass = pass_ok & (e["pass_end_location_x"] >= 102) & e["pass_end_location_y"].between(18, 62)

    # Carries
    is_carry = typ == "Carry"
    carry_gain = e["carry_end_location_x"] - x
    prog_carry = is_carry & (carry_gain >= 10) & (x >= 48)

    # Shots
    is_shot = typ == "Shot"
    np_shot = is_shot & (e["shot_type"] != "Penalty")
    npxg = e.loc[np_shot].groupby("player_id")["shot_statsbomb_xg"].sum()

    # Duels / defending
    tackle = (typ == "Duel") & (e["duel_type"] == "Tackle")
    aerial_won = (
        e["pass_aerial_won"].notna() | e["clearance_aerial_won"].notna()
        | e["shot_aerial_won"].notna() | e["miscontrol_aerial_won"].notna()
    )
    aerial_lost = (typ == "Duel") & (e["duel_type"] == "Aerial Lost")

    cols = {
        "passes": cnt(is_pass),
        "passes_completed": cnt(pass_ok),
        "progressive_passes": cnt(prog_pass),
        "passes_into_box": cnt(box_pass),
        "key_passes": cnt(is_pass & e["pass_shot_assist"].notna()),
        "assists": cnt(is_pass & e["pass_goal_assist"].notna()),
        "crosses": cnt(is_pass & e["pass_cross"].notna()),
        "through_balls": cnt(is_pass & e["pass_through_ball"].notna()),
        "carries": cnt(is_carry),
        "progressive_carries": cnt(prog_carry),
        "shots": cnt(np_shot),
        "np_goals": cnt(np_shot & (e["shot_outcome"] == "Goal")),
        "np_xg": npxg,
        "dribbles": cnt(typ == "Dribble"),
        "dribbles_completed": cnt((typ == "Dribble") & (e["dribble_outcome"] == "Complete")),
        "tackles": cnt(tackle),
        "tackles_won": cnt(tackle & e["duel_outcome"].isin(WON)),
        "interceptions": cnt(typ == "Interception"),
        "blocks": cnt(typ == "Block"),
        "clearances": cnt(typ == "Clearance"),
        "pressures": cnt(typ == "Pressure"),
        "pressures_final_third": cnt((typ == "Pressure") & final_third),
        "ball_recoveries": cnt((typ == "Ball Recovery") & e["ball_recovery_recovery_failure"].isna()),
        "aerials_won": cnt(aerial_won),
        "aerials_lost": cnt(aerial_lost),
        "dispossessed": cnt(typ == "Dispossessed"),
        "miscontrols": cnt(typ == "Miscontrol"),
        "dribbled_past": cnt(typ == "Dribbled Past"),
        "fouls_committed": cnt(typ == "Foul Committed"),
        "fouls_won": cnt(typ == "Foul Won"),
        "receptions": cnt(typ == "Ball Receipt*"),
        "box_touches": cnt(in_box & typ.isin(["Ball Receipt*", "Carry", "Pass", "Shot", "Dribble"])),
    }
    df = pd.DataFrame(cols).fillna(0)
    df = info.join(main_pos).join(minutes).join(df, how="left").fillna({c: 0 for c in cols})
    df.index.name = "player_id"

    count_cols = [c for c in cols if c != "np_xg"] + ["np_xg"]
    per90 = df.copy()
    for c in count_cols:
        per90[c + "_p90"] = per90[c] / per90["minutes"] * 90
    per90["pass_pct"] = per90["passes_completed"] / per90["passes"].replace(0, np.nan)
    per90["dribble_pct"] = per90["dribbles_completed"] / per90["dribbles"].replace(0, np.nan)
    per90["tackle_pct"] = per90["tackles_won"] / per90["tackles"].replace(0, np.nan)
    per90["aerial_pct"] = per90["aerials_won"] / (per90["aerials_won"] + per90["aerials_lost"]).replace(0, np.nan)
    per90["np_xg_per_shot"] = per90["np_xg"] / per90["shots"].replace(0, np.nan)

    outfield = per90[(per90["pos_group"] != "GK") & (per90["minutes"] >= MIN_MINUTES)]
    return per90, outfield


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_players, outfield = build()
    all_players.to_csv(OUT / "player_season.csv")
    outfield.to_csv(OUT / "player_per90.csv")
    print(f"all players: {len(all_players)}, outfield >= {MIN_MINUTES} min: {len(outfield)}")
    print(outfield["pos_group"].value_counts().to_string())


if __name__ == "__main__":
    main()
