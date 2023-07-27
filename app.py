# Import packages
from dash import Dash, dash_table, dcc, html, Input, Output, ctx
import pandas as pd, numpy as np
import datetime, pytz

date_str_today = datetime.datetime.now(pytz.timezone('US/Pacific')).strftime("%Y-%m-%d")
date_str_yesterday = (datetime.datetime.now(pytz.timezone('US/Pacific')) - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

def read_df_odds_from_gcs(gcs_pkl_url):
    df = pd.read_pickle(gcs_pkl_url).rename(columns={"player_name": "batting_name"})[['game_id', 'game_date', 'team_away', 'team_home', 'batting_name', 'property', 'over_odds', 'over_line', 'under_odds', 'under_line']]
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['over_odds'] = df.over_odds.astype(np.int32)
    #df = df[df.over_line < 1.0]
    return df

df_live_odds_hits = read_df_odds_from_gcs("https://storage.googleapis.com/major-league-baseball-public/odds_data/df_odds_today_hits.pkl")
df_live_odds_strikeouts = read_df_odds_from_gcs("https://storage.googleapis.com/major-league-baseball-public/odds_data/df_odds_today_strikeouts.pkl")

df_odds_hits = read_df_odds_from_gcs('https://storage.googleapis.com/major-league-baseball-public/odds_data/df_odds_history_hits.pkl')
df_odds_strikeouts = read_df_odds_from_gcs('https://storage.googleapis.com/major-league-baseball-public/odds_data/df_odds_history_strikeouts.pkl')


_default_threshold = 0.75

def read_df_live_prediction_from_gcs(gcs_pkl_url):
    df = pd.read_pickle(gcs_pkl_url).rename(columns={'date': 'game_date'})
    df = df[['game_id', 'game_date', 'batting_name', "prediction_label", "prediction_score", "theo_odds"]]
    df = df.sort_values(['prediction_score'], ascending=False)
    return df

def read_df_history_prediction_from_gcs(gcs_pkl_url):
    df = pd.read_pickle(gcs_pkl_url)
    df = df[['game_id', 'game_date', 'batting_name', "property_name", "property_value", "prediction_label", "prediction_score", "theo_odds"]]
    df = df[df.game_date <= date_str_yesterday].sort_values(['game_date'], ascending=False)

    return df


# 1hits
df_live_prediction_1hits = read_df_live_prediction_from_gcs("https://storage.googleapis.com/major-league-baseball-public/update_data/df_live_prediction_batting_1hits_recorded.pkl")
df_live_prediction_1hits_odds = df_live_prediction_1hits.merge(df_live_odds_hits, on=["game_id", "batting_name"], how="left")
df_live_prediction_1hits_odds_high_score = df_live_prediction_1hits_odds[(df_live_prediction_1hits_odds.prediction_score > _default_threshold)]

df_prediction_1hits = read_df_history_prediction_from_gcs("https://storage.googleapis.com/major-league-baseball-public/update_data/df_history_prediction_batting_1hits_recorded.pkl")
df_prediction_1hits_odds = df_prediction_1hits.merge(df_odds_hits, on=["game_id", "batting_name"], how="left")
df_prediction_1hits_odds_high_score = df_prediction_1hits_odds[(df_prediction_1hits_odds.prediction_score > _default_threshold)]

# 2hits
df_live_prediction_2hits = read_df_live_prediction_from_gcs("https://storage.googleapis.com/major-league-baseball-public/update_data/df_live_prediction_batting_2hits_recorded.pkl")
df_live_prediction_2hits_odds = df_live_prediction_2hits.merge(df_live_odds_hits, on=["game_id", "batting_name"], how="left")
df_live_prediction_2hits_odds_high_score = df_live_prediction_2hits_odds[(df_live_prediction_2hits_odds.prediction_score > _default_threshold)]

df_prediction_2hits = read_df_history_prediction_from_gcs("https://storage.googleapis.com/major-league-baseball-public/update_data/df_history_prediction_batting_2hits_recorded.pkl")
df_prediction_2hits_odds = df_prediction_2hits.merge(df_odds_hits, on=["game_id", "batting_name"], how="left")
df_prediction_2hits_odds_high_score = df_prediction_2hits_odds[(df_prediction_2hits_odds.prediction_score > _default_threshold)]

# 1strikeouts
df_live_prediction_strikeouts = read_df_live_prediction_from_gcs("https://storage.googleapis.com/major-league-baseball-public/update_data/df_live_prediction_batting_1strikeOuts_recorded.pkl")
df_live_prediction_strikeouts_odds = df_live_prediction_strikeouts.merge(df_live_odds_strikeouts, on=["game_id", "batting_name"], how="left")
df_live_prediction_strikeouts_odds_high_score = df_live_prediction_strikeouts_odds[(df_live_prediction_strikeouts_odds.prediction_score > _default_threshold)]

df_prediction_strikeouts = read_df_history_prediction_from_gcs("https://storage.googleapis.com/major-league-baseball-public/update_data/df_history_prediction_batting_1strikeOuts_recorded.pkl")
df_prediction_strikeouts_odds = df_prediction_strikeouts.merge(df_odds_strikeouts, on=["game_id", "batting_name"], how="left")
df_prediction_strikeouts_odds_high_score = df_prediction_strikeouts_odds[(df_prediction_strikeouts_odds.prediction_score > _default_threshold)]

def get_confident_bets_description(df_prediction_odds, target_prediction_label, over_or_under, target_line, score_threshold=0.75):
    df_confident_prediction_odds = df_prediction_odds[
        df_prediction_odds["prediction_score"] >= score_threshold].dropna()
    # the prediction_label should be separatedly checked. higher score does not always lead to prediction label. (maybe the score stands for both labels).
    df_confident_prediction_odds = df_confident_prediction_odds[df_confident_prediction_odds["prediction_label"] == target_prediction_label]
    if over_or_under == "over":
        df_confident_prediction_odds_opposite_line = df_confident_prediction_odds[df_confident_prediction_odds.over_line != target_line]
        df_confident_prediction_odds = df_confident_prediction_odds[df_confident_prediction_odds.over_line == target_line]
    else:
        df_confident_prediction_odds_opposite_line = df_confident_prediction_odds[df_confident_prediction_odds.over_line != target_line]
        df_confident_prediction_odds = df_confident_prediction_odds[df_confident_prediction_odds.over_line == target_line]
    if len(df_confident_prediction_odds) == 0:
        return "empty"
    prodiction_successes = df_confident_prediction_odds["property_value"].sum()
    l = len(df_confident_prediction_odds)
    ideal_profit_neg_odds = np.divide(100.0, np.abs(df_confident_prediction_odds.over_odds))
    ideal_profit_pos_odds = np.divide(df_confident_prediction_odds.over_odds, 100)
    ideal_profit = np.where(df_confident_prediction_odds.over_odds < 0, ideal_profit_neg_odds, ideal_profit_pos_odds)
    ideal_reward = np.add(1.0, ideal_profit)
    profit = np.sum(np.multiply(df_confident_prediction_odds["property_value"], ideal_reward)) - l

    profit = round(profit, 2)
    success_ratio = round(1.0 * prodiction_successes / l, 3) if l > 0 else 0
    return f'excluded w/ line > 1.0: {len(df_confident_prediction_odds_opposite_line)}, success recorded ratio: {success_ratio} ({prodiction_successes} out of {l}), profit: {profit}'

# Initialize the app
app = Dash(__name__)
server = app.server

# App layout
app.layout = html.Div([
    html.Div(
        [
            "Threshold",
            dcc.Input(id="threshold", type="number", value=_default_threshold, step=0.05),
            "Keep null records",
            dcc.Checklist(
                ['null'],
                ['null'],
                id="keep_null"
            ),
            dcc.Checklist(
                ['all_lines'],
                ['all_lines'],
                id="all_lines"
            ),
        ],
        style={"width": 250},
    ),
    "1Hits",
    html.Div(children='Live Prediction'),
    dash_table.DataTable(id="live_table_1hits", data=df_live_prediction_1hits_odds_high_score.to_dict('records'), page_size=10),
    html.Div(children='Prediction History'),
    dash_table.DataTable(id="history_table_1hits", data=df_prediction_1hits_odds_high_score.to_dict('records'), page_size=10),
    html.Div(id='confident_bet_profit_1hits'),
    "2Hits",
    html.Div(children='Live Prediction'),
    dash_table.DataTable(id="live_table_2hits", data=df_live_prediction_2hits_odds_high_score.to_dict('records'), page_size=10),
    html.Div(children='Prediction History'),
    dash_table.DataTable(id="history_table_2hits", data=df_prediction_2hits_odds_high_score.to_dict('records'), page_size=10),
    html.Div(id='confident_bet_profit_2hits'),
    "1Strikeouts",
    html.Div(children='Live Prediction'),
    dash_table.DataTable(id="live_table_1strikeouts", data=df_live_prediction_strikeouts_odds_high_score.to_dict('records'), page_size=10),
    html.Div(children='Prediction History'),
    dash_table.DataTable(id="history_table_1strikeouts", data=df_prediction_strikeouts_odds_high_score.to_dict('records'), page_size=10),
    html.Div(id='confident_bet_profit_1strikeouts'),
    dcc.Interval(
        id='interval-component',
        interval=10 * 60 * 1000, # in milliseconds
        n_intervals=0
    )
])

@app.callback(
    Output("live_table_1hits", "data"), Input("threshold", "value"), Input("keep_null", "value"), Input("all_lines", "value")
)
def update_live_table_1hits(threshold, keep_null, all_lines):
    df_live_prediction_1hits_odds_high_score = df_live_prediction_1hits_odds[(df_live_prediction_1hits_odds.prediction_score > threshold)]
    if not keep_null:
        df_live_prediction_1hits_odds_high_score = df_live_prediction_1hits_odds_high_score.dropna()
    if not all_lines:
        df_live_prediction_1hits_odds_high_score = df_live_prediction_1hits_odds_high_score[df_live_prediction_1hits_odds_high_score.over_line < 1.0]
    return df_live_prediction_1hits_odds_high_score.to_dict("records")

@app.callback(
    Output("live_table_2hits", "data"), Input("threshold", "value"), Input("keep_null", "value"), Input("all_lines", "value")
)
def update_live_table_2hits(threshold, keep_null, all_lines):
    df_live_prediction_2hits_odds_high_score = df_live_prediction_2hits_odds[(df_live_prediction_2hits_odds.prediction_score > threshold)]
    if not keep_null:
        df_live_prediction_2hits_odds_high_score = df_live_prediction_2hits_odds_high_score.dropna()
    if not all_lines:
        df_live_prediction_2hits_odds_high_score = df_live_prediction_2hits_odds_high_score[df_live_prediction_2hits_odds_high_score.over_line > 1.0]
    return df_live_prediction_2hits_odds_high_score.to_dict("records")

@app.callback(
    Output("live_table_1strikeouts", "data"), Input("threshold", "value"), Input("keep_null", "value"), Input("all_lines", "value")
)
def update_live_table_1strikeout(threshold, keep_null, all_lines):
    df_live_prediction_strikeoutsodds_high_score = df_live_prediction_strikeouts_odds[(df_live_prediction_strikeouts_odds.prediction_score > threshold)]
    if not keep_null:
        df_live_prediction_strikeoutsodds_high_score = df_live_prediction_strikeoutsodds_high_score.dropna()
    if not all_lines:
        df_live_prediction_strikeoutsodds_high_score = df_live_prediction_strikeoutsodds_high_score[df_live_prediction_strikeoutsodds_high_score.over_line < 1.0]
    return df_live_prediction_strikeoutsodds_high_score.to_dict("records")

@app.callback(
    Output("history_table_1hits", "data"), Input("threshold", "value"), Input("keep_null", "value"), Input("all_lines", "value")
)
def update_history_table_1hits(threshold, keep_null, all_lines):
    df_prediction_hits_odds_high_score = df_prediction_1hits_odds[(df_prediction_1hits_odds.prediction_score > threshold)]
    if not keep_null:
        df_prediction_hits_odds_high_score = df_prediction_hits_odds_high_score.dropna()
    if not all_lines:
        df_prediction_hits_odds_high_score = df_prediction_hits_odds_high_score[df_prediction_hits_odds_high_score.over_line < 1.0]
    return df_prediction_hits_odds_high_score.to_dict("records")

@app.callback(
    Output("history_table_2hits", "data"), Input("threshold", "value"), Input("keep_null", "value"), Input("all_lines", "value")
)
def update_history_table_2hits(threshold, keep_null, all_lines):
    df_prediction_hits_odds_high_score = df_prediction_2hits_odds[(df_prediction_2hits_odds.prediction_score > threshold)]
    if not keep_null:
        df_prediction_hits_odds_high_score = df_prediction_hits_odds_high_score.dropna()
    if not all_lines:
        df_prediction_hits_odds_high_score = df_prediction_hits_odds_high_score[df_prediction_hits_odds_high_score.over_line > 1.0]
    return df_prediction_hits_odds_high_score.to_dict("records")

@app.callback(
    Output("history_table_1strikeouts", "data"), Input("threshold", "value"), Input("keep_null", "value"), Input("all_lines", "value")
)
def update_history_table_1strikeout(threshold, keep_null, all_lines):
    df_prediction_strikeouts_odds_high_score = df_prediction_strikeouts_odds[(df_prediction_strikeouts_odds.prediction_score > threshold)]
    if not keep_null:
        df_prediction_strikeouts_odds_high_score = df_prediction_strikeouts_odds_high_score.dropna()
    if not all_lines:
        df_prediction_strikeouts_odds_high_score = df_prediction_strikeouts_odds_high_score[df_prediction_strikeouts_odds_high_score.over_line < 1.0]
    return df_prediction_strikeouts_odds_high_score.to_dict("records")

@app.callback(
    Output(component_id='confident_bet_profit_1hits', component_property='children'),
    Input(component_id='threshold', component_property='value')
)
def update_confident_1hits_bet_profit(threshold):
    return get_confident_bets_description(df_prediction_1hits_odds, target_prediction_label=1.0, over_or_under="over", target_line=0.5, score_threshold=threshold)

@app.callback(
    Output(component_id='confident_bet_profit_2hits', component_property='children'),
    Input(component_id='threshold', component_property='value')
)
def update_confident_2hits_bet_profit(threshold):
    return get_confident_bets_description(df_prediction_2hits_odds, target_prediction_label=0.0, over_or_under="under", target_line=1.5, score_threshold=threshold)

@app.callback(
    Output(component_id='confident_bet_profit_1strikeouts', component_property='children'),
    Input(component_id='threshold', component_property='value')
)
def update_confident_1strikeouts_bet_profit(threshold):
    return get_confident_bets_description(df_prediction_strikeouts_odds, target_prediction_label=1.0, over_or_under="over", target_line=0.5, score_threshold=threshold)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

