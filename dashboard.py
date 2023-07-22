# Import packages
from dash import Dash, dash_table, dcc, html, Input, Output, ctx
import pandas as pd, numpy as np
import datetime, pytz

date_str_today = datetime.datetime.now(pytz.timezone('US/Pacific')).strftime("%Y-%m-%d")
date_str_yesterday = (datetime.datetime.now(pytz.timezone('US/Pacific')) - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

df_odds = pd.read_pickle('odds_data/df_odds.pkl').rename(columns={"player_name": "batting_name"})[['game_id', 'game_date', 'team_away', 'team_home', 'batting_name', 'property', 'over_odds', 'over_line']]
df_odds_1hits = df_odds[(df_odds.property=="Hits") & (df_odds.over_line < 1.0)]
df_odds_1strikeouts = df_odds[(df_odds.property=="Strikeouts") & (df_odds.over_line < 1.0)]

_default_threshold = 0.75

# 1hits
df_live_prediction_1hits = pd.read_pickle('update_data/temp/df_live_prediction_batting_1hits_recorded_2023-07-20.pkl')[['game_id', 'game_date', 'batting_shortName', 'batting_name', "prediction_score", "theo_odds"]]
df_live_prediction_1hits = df_live_prediction_1hits.sort_values(['prediction_score'], ascending=False)
df_live_prediction_1hits_odds = df_live_prediction_1hits.merge(df_odds_1hits, on=["game_id", "batting_name"], how="left")
df_live_prediction_1hits_odds_high_score = df_live_prediction_1hits_odds[(df_live_prediction_1hits_odds.prediction_score > _default_threshold)]

df_prediction_1hits = pd.read_pickle('update_data/temp/df_prediction_batting_1hits_recorded_2023-07-20.pkl')
df_prediction_1hits = df_prediction_1hits[df_prediction_1hits.game_date <= date_str_yesterday].sort_values(['game_date'], ascending=False)
df_prediction_1hits = df_prediction_1hits[['game_id', 'game_date', 'batting_shortName', 'batting_name', "batting_1hits_recorded", "prediction_label", "prediction_score", "theo_odds"]]
df_prediction_1hits_odds = df_prediction_1hits.merge(df_odds_1hits, on=["game_id", "batting_name"], how="left")
df_prediction_1hits_odds_high_score = df_prediction_1hits_odds[(df_prediction_1hits_odds.prediction_score > _default_threshold)]

# 1strikeouts
df_live_prediction_1strikeouts = pd.read_pickle('update_data/temp/df_live_prediction_batting_1strikeOuts_recorded_2023-07-20.pkl')[['game_id', 'game_date', 'batting_shortName', 'batting_name', "prediction_score", "theo_odds"]]
df_live_prediction_1strikeouts = df_live_prediction_1strikeouts.sort_values(['prediction_score'], ascending=False)
df_live_prediction_1strikeouts_odds = df_live_prediction_1strikeouts.merge(df_odds_1strikeouts, on=["game_id", "batting_name"], how="left")
df_live_prediction_1strikeouts_odds_high_score = df_live_prediction_1strikeouts_odds[(df_live_prediction_1strikeouts_odds.prediction_score > _default_threshold)]

df_prediction_1strikeouts = pd.read_pickle('update_data/temp/df_prediction_batting_1strikeOuts_recorded_2023-07-21.pkl')
df_prediction_1strikeouts = df_prediction_1strikeouts[df_prediction_1strikeouts.game_date <= date_str_yesterday].sort_values(['game_date'], ascending=False)
df_prediction_1strikeouts = df_prediction_1strikeouts[['game_id', 'game_date', 'batting_shortName', 'batting_name', "batting_1strikeOuts_recorded", "prediction_label", "prediction_score", "theo_odds"]]
df_prediction_1strikeouts_odds = df_prediction_1strikeouts.merge(df_odds_1strikeouts, on=["game_id", "batting_name"], how="left")
df_prediction_1strikeouts_odds_high_score = df_prediction_1strikeouts_odds[(df_prediction_1strikeouts_odds.prediction_score > _default_threshold)]

def get_confident_bets(df_prediction_odds, property_name, score_threshold=0.75):
    df_confident_prediction_odds = df_prediction_odds[
        df_prediction_odds["prediction_score"] >= score_threshold].dropna()
    # the prediction_label should be separatedly checked. higher score does not always lead to prediction label. (maybe the score stands for both labels).
    df_confident_prediction_odds = df_confident_prediction_odds[df_confident_prediction_odds["prediction_label"] == 1]
    if len(df_confident_prediction_odds) == 0:
        return 0, 0, 0
    prodiction_successes = df_confident_prediction_odds[property_name].sum()
    l = len(df_confident_prediction_odds)
    df_confident_prediction_odds = df_confident_prediction_odds.loc[:, ~df_confident_prediction_odds.columns.duplicated()].copy()
    ideal_reward = np.add(1.0, np.divide(100.0, np.abs(df_confident_prediction_odds.over_odds)))
    profit = np.sum(np.multiply(df_confident_prediction_odds[property_name], ideal_reward)) - l

    return prodiction_successes, l, profit

# Initialize the app
app = Dash(__name__)

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
    dash_table.DataTable(id="history_table_1hits", data=df_prediction_1hits_odds_high_score.to_dict('records'), page_size=10),
    html.Div(id='confident_bet_profit_1hits'),
    "1Strikeouts",
    html.Div(children='Live Prediction'),
    dash_table.DataTable(id="live_table_1strikeouts", data=df_live_prediction_1strikeouts_odds_high_score.to_dict('records'), page_size=10),
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
    df_live_prediction_1strikeoutsodds_high_score = df_live_prediction_1strikeouts_odds[(df_live_prediction_1strikeouts_odds.prediction_score > threshold)]
    return df_live_prediction_1strikeoutsodds_high_score.to_dict("records")

@app.callback(
    Output("history_table_1hits", "data"), Input("threshold", "value")
)
def update_table_1hits(threshold):
    df_prediction_1hits_odds_high_score = df_prediction_1hits_odds[(df_prediction_1hits_odds.prediction_score > threshold)]
    return df_prediction_1hits_odds_high_score.to_dict("records")

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
    prodiction_successes, l, profit = get_confident_bets(df_prediction_1hits_odds, "batting_1hits_recorded", score_threshold=threshold)
    prodiction_successes, profit = prodiction_successes, round(profit, 2)
    success_ratio = round(1.0 * prodiction_successes / l, 3) if l > 0 else 0
    return f'success recorded ratio: {success_ratio} ({prodiction_successes} out of {l}), profit: {profit}'

@app.callback(
    Output(component_id='confident_bet_profit_1strikeouts', component_property='children'),
    Input(component_id='threshold', component_property='value')
)
def update_confident_1strikeouts_bet_profit(threshold):
    prodiction_successes, l, profit = get_confident_bets(df_prediction_1strikeouts_odds, "batting_1strikeOuts_recorded", score_threshold=threshold)
    prodiction_successes, profit = prodiction_successes, round(profit, 2)
    success_ratio = round(1.0 * prodiction_successes / l, 3) if l > 0 else 0
    return f'success recorded ratio: {success_ratio} ({prodiction_successes} out of {l}), profit: {profit}'

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

