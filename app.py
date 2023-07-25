# Import packages
from dash import Dash, dash_table, dcc, html, Input, Output, ctx
import pandas as pd, numpy as np
import datetime, pytz

date_str_today = datetime.datetime.now(pytz.timezone('US/Pacific')).strftime("%Y-%m-%d")
date_str_yesterday = (datetime.datetime.now(pytz.timezone('US/Pacific')) - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

#df_live_odds = pd.read_pickle('odds_data/df_odds_today_all.pkl').rename(columns={"player_name": "batting_name"})[['game_id', 'game_date', 'team_away', 'team_home', 'batting_name', 'property', 'over_odds', 'over_line']]
#df_live_odds_1hits = df_live_odds[(df_live_odds.property=="Hits") & (df_live_odds.over_line < 1.0)]
#df_live_odds_1strikeouts = df_live_odds[(df_live_odds.property=="Strikeouts") & (df_live_odds.over_line < 1.0)]

def read_df_odds_from_gcs(gcs_pkl_url):
    df = pd.read_pickle(gcs_pkl_url).rename(columns={"player_name": "batting_name"})[['game_id', 'game_date', 'team_away', 'team_home', 'batting_name', 'property', 'over_odds', 'over_line']]
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['over_odds'] = df.over_odds.astype(np.int32)
    #df = df[df.over_line < 1.0]
    return df

df_live_odds_1hits = read_df_odds_from_gcs("https://storage.googleapis.com/major-league-baseball-public/odds_data/df_odds_today_hits.pkl")
df_live_odds_1strikeouts = read_df_odds_from_gcs("https://storage.googleapis.com/major-league-baseball-public/odds_data/df_odds_today_strikeouts.pkl")

#df_odds = pd.read_pickle('odds_data/df_odds_2023_all.pkl').rename(columns={"player_name": "batting_name"})[['game_id', 'game_date', 'team_away', 'team_home', 'batting_name', 'property', 'over_odds', 'over_line']]
#df_odds_1hits = df_odds[(df_odds.property=="Hits") & (df_odds.over_line < 1.0)]
#df_odds_1strikeouts = df_odds[(df_odds.property=="Strikeouts") & (df_odds.over_line < 1.0)]
df_odds_hits = pd.read_pickle('odds_data/df_odds_history_hits.pkl').rename(columns={"player_name": "batting_name"})[['game_id', 'game_date', 'team_away', 'team_home', 'batting_name', 'property', 'over_odds', 'over_line']]
df_odds_1hits = df_odds_hits[df_odds_hits.over_line < 1]
df_odds_strikeouts = pd.read_pickle('odds_data/df_odds_history_strikeouts.pkl').rename(columns={"player_name": "batting_name"})[['game_id', 'game_date', 'team_away', 'team_home', 'batting_name', 'property', 'over_odds', 'over_line']]
df_odds_1strikeouts = df_odds_strikeouts[df_odds_strikeouts.over_line < 1]


_default_threshold = 0.75

def read_df_live_prediction_from_gcs(gcs_pkl_url):
    df = pd.read_pickle(gcs_pkl_url).rename(columns={'date': 'game_date'})
    df = df[['game_id', 'game_date', 'batting_shortName', 'batting_name', "prediction_score", "theo_odds"]]
    return df

def read_df_history_prediction_from_gcs(gcs_pkl_url):
    df = pd.read_pickle(gcs_pkl_url)
    df = df[['game_id', 'game_date', 'batting_shortName', 'batting_name', "prediction_score", "theo_odds"]]
    return df


# 1hits
#df_live_prediction_1hits = pd.read_pickle('update_data/df_live_prediction_batting_1hits_recorded.pkl')[['game_id', 'date', 'batting_shortName', 'batting_name', "prediction_score", "theo_odds"]]
df_live_prediction_1hits = read_df_live_prediction_from_gcs("https://storage.googleapis.com/major-league-baseball-public/update_data/df_live_prediction_batting_1hits_recorded.pkl")
df_live_prediction_1hits = df_live_prediction_1hits.sort_values(['prediction_score'], ascending=False)
df_live_prediction_1hits_odds = df_live_prediction_1hits.merge(df_live_odds_1hits, on=["game_id", "batting_name"], how="left")
df_live_prediction_1hits_odds_high_score = df_live_prediction_1hits_odds[(df_live_prediction_1hits_odds.prediction_score > _default_threshold)]

#df_prediction_hits = pd.read_pickle('update_data/temp/df_prediction_batting_1hits_recorded_2023-07-20.pkl')
df_prediction_hits = pd.read_pickle('update_data/df_history_prediction_batting_1hits_recorded.pkl').rename(columns={'date': 'game_date'})
df_prediction_hits = df_prediction_hits[df_prediction_hits.game_date <= date_str_yesterday].sort_values(['game_date'], ascending=False)
df_prediction_hits = df_prediction_hits[['game_id', 'game_date', 'batting_name', "property_name", "property_value", "prediction_label", "prediction_score", "theo_odds"]]
df_prediction_hits_odds = df_prediction_hits.merge(df_odds_1hits, on=["game_id", "batting_name"], how="left")
df_prediction_hits_odds_high_score = df_prediction_hits_odds[(df_prediction_hits_odds.prediction_score > _default_threshold)]

# 1strikeouts
#df_live_prediction_strikeouts = pd.read_pickle('update_data/df_live_prediction_batting_1strikeOuts_recorded.pkl')[['game_id', 'date', 'batting_shortName', 'batting_name', "prediction_score", "theo_odds"]]
df_live_prediction_strikeouts = read_df_live_prediction_from_gcs("https://storage.googleapis.com/major-league-baseball-public/update_data/df_live_prediction_batting_1strikeOuts_recorded.pkl")
df_live_prediction_strikeouts = df_live_prediction_strikeouts.sort_values(['prediction_score'], ascending=False)
df_live_prediction_strikeouts_odds = df_live_prediction_strikeouts.merge(df_live_odds_1strikeouts, on=["game_id", "batting_name"], how="left")
df_live_prediction_strikeouts_odds_high_score = df_live_prediction_strikeouts_odds[(df_live_prediction_strikeouts_odds.prediction_score > _default_threshold)]

#df_prediction_1strikeouts = pd.read_pickle('update_data/temp/df_prediction_batting_1strikeOuts_recorded_2023-07-21.pkl')
df_prediction_1strikeouts = pd.read_pickle('update_data/df_history_prediction_batting_1strikeOuts_recorded.pkl').rename(columns={'date': 'game_date'})
df_prediction_1strikeouts = df_prediction_1strikeouts[df_prediction_1strikeouts.game_date <= date_str_yesterday].sort_values(['game_date'], ascending=False)
df_prediction_1strikeouts = df_prediction_1strikeouts[['game_id', 'game_date', 'batting_name', "property_name", "property_value", "prediction_label", "prediction_score", "theo_odds"]]
df_prediction_1strikeouts_odds = df_prediction_1strikeouts.merge(df_odds_1strikeouts, on=["game_id", "batting_name"], how="left")
df_prediction_1strikeouts_odds_high_score = df_prediction_1strikeouts_odds[(df_prediction_1strikeouts_odds.prediction_score > _default_threshold)]

def get_confident_bets_description(df_prediction_odds, score_threshold=0.75):
    df_confident_prediction_odds = df_prediction_odds[
        df_prediction_odds["prediction_score"] >= score_threshold].dropna()
    # the prediction_label should be separatedly checked. higher score does not always lead to prediction label. (maybe the score stands for both labels).
    df_confident_prediction_odds = df_confident_prediction_odds[df_confident_prediction_odds["prediction_label"] == 1]
    df_confident_prediction_odds_over_line_gt_1 = df_confident_prediction_odds[df_confident_prediction_odds.over_line > 1.0]
    df_confident_prediction_odds = df_confident_prediction_odds[df_confident_prediction_odds.over_line < 1.0]
    if len(df_confident_prediction_odds) == 0:
        return 0, 0, 0
    prodiction_successes = df_confident_prediction_odds["property_value"].sum()
    l = len(df_confident_prediction_odds)
    df_confident_prediction_odds = df_confident_prediction_odds.loc[:, ~df_confident_prediction_odds.columns.duplicated()].copy()
    ideal_reward = np.add(1.0, np.divide(100.0, np.abs(df_confident_prediction_odds.over_odds)))
    profit = np.sum(np.multiply(df_confident_prediction_odds["property_value"], ideal_reward)) - l

    prodiction_successes, profit = prodiction_successes, round(profit, 2)
    success_ratio = round(1.0 * prodiction_successes / l, 3) if l > 0 else 0
    return f'excluded w/ line > 1.0: {len(df_confident_prediction_odds_over_line_gt_1)}, success recorded ratio: {success_ratio} ({prodiction_successes} out of {l}), profit: {profit}'

# Initialize the app
app = Dash(__name__)
server = app.server

# App layout
app.layout = html.Div([
    html.Div(
        [
            "Threshold",
            dcc.Input(id="threshold", type="number", value=_default_threshold, step=0.05),
        ],
        style={"width": 250},
    ),
    "1Hits",
    html.Div(children='Live Prediction'),
    dash_table.DataTable(id="live_table_1hits", data=df_live_prediction_1hits_odds_high_score.to_dict('records'), page_size=10),
    html.Div(children='Prediction History'),
    dash_table.DataTable(id="history_table_1hits", data=df_prediction_hits_odds_high_score.to_dict('records'), page_size=10),
    html.Div(id='confident_bet_profit_1hits'),
    "1Strikeouts",
    html.Div(children='Live Prediction'),
    dash_table.DataTable(id="live_table_1strikeouts", data=df_live_prediction_strikeouts_odds_high_score.to_dict('records'), page_size=10),
    html.Div(children='Prediction History'),
    dash_table.DataTable(id="history_table_1strikeouts", data=df_prediction_1strikeouts_odds_high_score.to_dict('records'), page_size=10),
    html.Div(id='confident_bet_profit_1strikeouts'),
])

@app.callback(
    Output("live_table_1hits", "data"), Input("threshold", "value")
)
def update_table_1hits(threshold):
    df_live_prediction_1hits_odds_high_score = df_live_prediction_1hits_odds[(df_live_prediction_1hits_odds.prediction_score > threshold)]
    return df_live_prediction_1hits_odds_high_score.to_dict("records")

@app.callback(
    Output("live_table_1strikeouts", "data"), Input("threshold", "value")
)
def update_table_1strikeout(threshold):
    df_live_prediction_strikeoutsodds_high_score = df_live_prediction_strikeouts_odds[(df_live_prediction_strikeouts_odds.prediction_score > threshold)]
    return df_live_prediction_strikeoutsodds_high_score.to_dict("records")

@app.callback(
    Output("history_table_1hits", "data"), Input("threshold", "value")
)
def update_table_1hits(threshold):
    df_prediction_hits_odds_high_score = df_prediction_hits_odds[(df_prediction_hits_odds.prediction_score > threshold)]
    return df_prediction_hits_odds_high_score.to_dict("records")

@app.callback(
    Output("history_table_1strikeouts", "data"), Input("threshold", "value")
)
def update_table_1strikeout(threshold):
    df_prediction_1strikeouts_odds_high_score = df_prediction_1strikeouts_odds[(df_prediction_1strikeouts_odds.prediction_score > threshold)]
    return df_prediction_1strikeouts_odds_high_score.to_dict("records")

@app.callback(
    Output(component_id='confident_bet_profit_1hits', component_property='children'),
    Input(component_id='threshold', component_property='value')
)
def update_confident_1hits_bet_profit(threshold):
    return get_confident_bets_description(df_prediction_hits_odds, score_threshold=threshold)

@app.callback(
    Output(component_id='confident_bet_profit_1strikeouts', component_property='children'),
    Input(component_id='threshold', component_property='value')
)
def update_confident_1strikeouts_bet_profit(threshold):
    return get_confident_bets_description(df_prediction_1strikeouts_odds, score_threshold=threshold)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

