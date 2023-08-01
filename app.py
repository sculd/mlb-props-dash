# Import packages
from dash import Dash, dash_table, dcc, html, Input, Output, ctx
import pandas as pd, numpy as np
import datetime, pytz

from flask_caching import Cache

date_str_today = datetime.datetime.now(pytz.timezone('US/Pacific')).strftime("%Y-%m-%d")
date_str_yesterday = (datetime.datetime.now(pytz.timezone('US/Pacific')) - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

_default_threshold = 0.75

GCS_URL_LIVE_PREDICTION_1HITS = "https://storage.googleapis.com/major-league-baseball-public/update_data/df_live_prediction_batting_1hits_recorded.pkl"
GCS_URL_LIVE_PREDICTION_2HITS = "https://storage.googleapis.com/major-league-baseball-public/update_data/df_live_prediction_batting_2hits_recorded.pkl"
GCS_URL_LIVE_PREDICTION_1STRIKEOUT = "https://storage.googleapis.com/major-league-baseball-public/update_data/df_live_prediction_batting_1strikeOuts_recorded.pkl"
GCS_URL_LIVE_ODDS_HITS = "https://storage.googleapis.com/major-league-baseball-public/odds_data/df_odds_today_hits.pkl"
GCS_URL_LIVE_ODDS_STRIKEOUTS = "https://storage.googleapis.com/major-league-baseball-public/odds_data/df_odds_today_strikeouts.pkl"

GCS_URL_HISTORY_PREDICTION_1HITS = "https://storage.googleapis.com/major-league-baseball-public/update_data/df_history_prediction_batting_1hits_recorded.pkl"
GCS_URL_HISTORY_PREDICTION_2HITS = "https://storage.googleapis.com/major-league-baseball-public/update_data/df_history_prediction_batting_2hits_recorded.pkl"
GCS_URL_HISTORY_PREDICTION_1STRIKEOUT = "https://storage.googleapis.com/major-league-baseball-public/update_data/df_history_prediction_batting_1strikeOuts_recorded.pkl"
GCS_URL_HISTORY_ODDS_HITS = "https://storage.googleapis.com/major-league-baseball-public/odds_data/df_odds_history_hits.pkl"
GCS_URL_HISTORY_ODDS_STRIKEOUTS = "https://storage.googleapis.com/major-league-baseball-public/odds_data/df_odds_history_strikeouts.pkl"

# Initialize the app
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
cache = Cache(app.server, config={
    'CACHE_TYPE': 'SimpleCache',
})

@cache.memoize(timeout=60*30) # in seconds
def read_df_pkl_as_dict_from_gcs(gcs_pkl_url):
    df = pd.read_pickle(gcs_pkl_url).rename(columns={'date': 'game_date'})
    return df.to_dict()

@cache.memoize(timeout=60*10) # in seconds
def read_df_odds_from_gcs(gcs_pkl_url):
    #df = pd.read_pickle(gcs_pkl_url)
    df = pd.DataFrame.from_dict(read_df_pkl_as_dict_from_gcs(gcs_pkl_url))
    df = df.rename(columns={"player_name": "batting_name"})
    df = df[['game_id', 'game_date', 'team_away', 'team_home', 'batting_name', 'property', 'over_odds', 'over_line', 'under_odds', 'under_line']]
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['over_odds'] = df.over_odds.astype(np.int32)
    #df = df[df.over_line < 1.0]
    return df

@cache.memoize(timeout=60*10) # in seconds
def read_df_live_prediction_from_gcs(gcs_pkl_url):
    #df = pd.read_pickle(gcs_pkl_url)
    df = pd.DataFrame.from_dict(read_df_pkl_as_dict_from_gcs(gcs_pkl_url))
    df = df.rename(columns={'date': 'game_date'})
    df = df[['game_id', 'game_date', 'batting_name', "prediction_label", "prediction_score", "theo_odds"]]
    df = df.sort_values(['prediction_score'], ascending=False)
    return df

@cache.memoize(timeout=60*10) # in seconds
def read_df_history_prediction_from_gcs(gcs_pkl_url):
    #df = pd.read_pickle(gcs_pkl_url)
    df = pd.DataFrame.from_dict(read_df_pkl_as_dict_from_gcs(gcs_pkl_url))
    df = df[['game_id', 'game_date', 'batting_name', "property_name", "property_value", "prediction_label", "prediction_score", "theo_odds"]]
    df = df[df.game_date <= date_str_yesterday].sort_values(['game_date'], ascending=False)

    return df

def get_confident_bets_description(df_prediction_odds, target_prediction_label, over_or_under, target_line, score_threshold=0.75):
    df_confident_prediction_odds = df_prediction_odds[
        df_prediction_odds["prediction_score"] >= score_threshold].dropna()
    # the prediction_label should be separatedly checked. higher score does not always lead to prediction label. (maybe the score stands for both labels).
    df_confident_prediction_odds = df_confident_prediction_odds[df_confident_prediction_odds["prediction_label"] == target_prediction_label]
    if over_or_under == "over":
        df_confident_prediction_odds_opposite_line = df_confident_prediction_odds[df_confident_prediction_odds.over_line != target_line]
        df_confident_prediction_odds = df_confident_prediction_odds[df_confident_prediction_odds.over_line == target_line]
    else:
        df_confident_prediction_odds_opposite_line = df_confident_prediction_odds[df_confident_prediction_odds.under_line != target_line]
        df_confident_prediction_odds = df_confident_prediction_odds[df_confident_prediction_odds.under_line == target_line]
    if len(df_confident_prediction_odds) == 0:
        return "empty"
    prodiction_successes = len(df_confident_prediction_odds[df_confident_prediction_odds.property_value == target_prediction_label])
    l = len(df_confident_prediction_odds)
    if over_or_under == "over":
        ideal_profit_neg_odds = np.divide(100.0, np.abs(df_confident_prediction_odds.over_odds))
        ideal_profit_pos_odds = np.divide(df_confident_prediction_odds.over_odds, 100)
        ideal_profit = np.where(df_confident_prediction_odds.over_odds < 0, ideal_profit_neg_odds, ideal_profit_pos_odds)
    else:
        ideal_profit_neg_odds = np.divide(100.0, np.abs(df_confident_prediction_odds.under_odds))
        ideal_profit_pos_odds = np.divide(df_confident_prediction_odds.under_odds, 100)
        ideal_profit = np.where(df_confident_prediction_odds.under_odds < 0, ideal_profit_neg_odds, ideal_profit_pos_odds)
    ideal_reward = np.add(1.0, ideal_profit)
    profit = np.sum(np.multiply(np.where(df_confident_prediction_odds.property_value == target_prediction_label, 1, 0), ideal_reward)) - l

    profit = round(profit, 2)
    success_ratio = round(1.0 * prodiction_successes / l, 3) if l > 0 else 0
    return f'excluded different line than {target_line}: {len(df_confident_prediction_odds_opposite_line)}, success recorded ratio: {success_ratio} ({prodiction_successes} out of {l}), profit: {profit}'

# App layout
app.layout = html.Div([
    html.Div(
        [
            "Threshold",
            dcc.Input(id="threshold", type="number", value=_default_threshold, step=0.025),
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
    dcc.Tabs(id='property-tabs', value='1Hits', children=[
        dcc.Tab(label='1Hits', value='1Hits'),
        dcc.Tab(label='2Hits', value='2Hits'),
        dcc.Tab(label='1Strikeouts', value='1Strikeouts'),
    ]),
    html.Div(id='property-tab-content'),
])


@app.callback(
    Output('property-tab-content', 'children'),
    Input('property-tabs', 'value')
)
def render_content(tab):
    if tab == '1Hits':
        return html.Div([
            html.Div(children='Live Prediction'),
            dash_table.DataTable(id="live_table_1hits", data=get_live_data(GCS_URL_LIVE_PREDICTION_1HITS, GCS_URL_LIVE_ODDS_HITS, _default_threshold, keep_null=True, all_lines=True), page_size=10),
            html.Div(children='Prediction History'),
            dash_table.DataTable(id="history_table_1hits", data=get_history_data(GCS_URL_HISTORY_PREDICTION_1HITS, GCS_URL_HISTORY_ODDS_HITS, _default_threshold, keep_null=True, all_lines=True), page_size=10),
            html.Div(id='confident_over_bet_profit_1hits'),
            html.Div(id='confident_under_bet_profit_1hits'),
        ])
    elif tab == '2Hits':
        return html.Div([
            html.Div(children='Live Prediction'),
            dash_table.DataTable(id="live_table_2hits", data=get_live_data(GCS_URL_LIVE_PREDICTION_2HITS, GCS_URL_LIVE_ODDS_HITS, _default_threshold, keep_null=True, all_lines=True), page_size=10),
            html.Div(children='Prediction History'),
            dash_table.DataTable(id="history_table_2hits", data=get_history_data(GCS_URL_HISTORY_PREDICTION_2HITS, GCS_URL_HISTORY_ODDS_HITS, _default_threshold, keep_null=True, all_lines=True), page_size=10),
            html.Div(id='confident_under_bet_profit_2hits'),
        ])
    elif tab == '1Strikeouts':
        return html.Div([
            html.Div(children='Live Prediction'),
            dash_table.DataTable(id="live_table_1strikeouts", data=get_live_data(GCS_URL_LIVE_PREDICTION_1STRIKEOUT, GCS_URL_LIVE_ODDS_STRIKEOUTS, _default_threshold, keep_null=True, all_lines=True), page_size=10),
            html.Div(children='Prediction History'),
            dash_table.DataTable(id="history_table_1strikeouts", data=get_history_data(GCS_URL_HISTORY_PREDICTION_1STRIKEOUT, GCS_URL_HISTORY_ODDS_STRIKEOUTS, _default_threshold, keep_null=True, all_lines=True), page_size=10),
            html.Div(id='confident_over_bet_profit_1strikeouts'),
            html.Div(id='confident_under_bet_profit_1strikeouts'),
        ])

def merge_prediction_odds(df_prediction, df_odds):
    return df_prediction.merge(df_odds, on=["game_id", "batting_name"], how="left")

def get_live_data(live_prediction_gcs, live_odds_gcs, threshold, keep_null, all_lines):
    df_prediction = read_df_live_prediction_from_gcs(live_prediction_gcs)
    df_odds = read_df_odds_from_gcs(live_odds_gcs)
    df_prediction_odds = merge_prediction_odds(df_prediction, df_odds)
    df_prediction_odds_high_score = df_prediction_odds[(df_prediction_odds.prediction_score > threshold)]
    if not keep_null:
        df_prediction_odds_high_score = df_prediction_odds_high_score.dropna()
    if not all_lines:
        df_prediction_odds_high_score = df_prediction_odds_high_score[df_prediction_odds_high_score.over_line < 1.0]
    return df_prediction_odds_high_score.to_dict("records")

def get_history_data(history_prediction_gcs, history_odds_gcs, threshold, keep_null, all_lines):
    df_prediction = read_df_history_prediction_from_gcs(history_prediction_gcs)
    df_odds = read_df_odds_from_gcs(history_odds_gcs)
    df_prediction_odds = merge_prediction_odds(df_prediction, df_odds)
    df_prediction_odds_high_score = df_prediction_odds[(df_prediction_odds.prediction_score > threshold)]
    if not keep_null:
        df_prediction_odds_high_score = df_prediction_odds_high_score.dropna()
    if not all_lines:
        df_prediction_odds_high_score = df_prediction_odds_high_score[df_prediction_odds_high_score.over_line < 1.0]
    return df_prediction_odds_high_score.to_dict("records")

@app.callback(
    Output("live_table_1hits", "data"), Input("threshold", "value"), Input("keep_null", "value"), Input("all_lines", "value")
)
def update_live_table_1hits(threshold, keep_null, all_lines):
    return get_live_data(
        GCS_URL_LIVE_PREDICTION_1HITS,
        GCS_URL_LIVE_ODDS_HITS,
        threshold, keep_null, all_lines)

@app.callback(
    Output("live_table_2hits", "data"), Input("threshold", "value"), Input("keep_null", "value"), Input("all_lines", "value")
)
def update_live_table_2hits(threshold, keep_null, all_lines):
    return get_live_data(
        GCS_URL_LIVE_PREDICTION_2HITS,
        GCS_URL_LIVE_ODDS_HITS,
        threshold, keep_null, all_lines)

@app.callback(
    Output("live_table_1strikeouts", "data"), Input("threshold", "value"), Input("keep_null", "value"), Input("all_lines", "value")
)
def update_live_table_1strikeout(threshold, keep_null, all_lines):
    return get_live_data(
        GCS_URL_LIVE_PREDICTION_1STRIKEOUT,
        GCS_URL_LIVE_ODDS_STRIKEOUTS,
        threshold, keep_null, all_lines)

@app.callback(
    Output("history_table_1hits", "data"), Input("threshold", "value"), Input("keep_null", "value"), Input("all_lines", "value")
)
def update_history_table_1hits(threshold, keep_null, all_lines):
    return get_history_data(
        GCS_URL_HISTORY_PREDICTION_1HITS,
        GCS_URL_HISTORY_ODDS_HITS,
        threshold, keep_null, all_lines)

@app.callback(
    Output("history_table_2hits", "data"), Input("threshold", "value"), Input("keep_null", "value"), Input("all_lines", "value")
)
def update_history_table_2hits(threshold, keep_null, all_lines):
    return get_history_data(
        GCS_URL_HISTORY_PREDICTION_2HITS,
        GCS_URL_HISTORY_ODDS_HITS,
        threshold, keep_null, all_lines)

@app.callback(
    Output("history_table_1strikeouts", "data"), Input("threshold", "value"), Input("keep_null", "value"), Input("all_lines", "value")
)
def update_history_table_1strikeout(threshold, keep_null, all_lines):
    return get_history_data(
        GCS_URL_HISTORY_PREDICTION_1STRIKEOUT,
        GCS_URL_HISTORY_ODDS_STRIKEOUTS,
        threshold, keep_null, all_lines)

def get_df_history_prediction_odds_1hits_odds():
    df_prediction = read_df_history_prediction_from_gcs(GCS_URL_HISTORY_PREDICTION_1HITS)
    df_odds = read_df_odds_from_gcs("https://storage.googleapis.com/major-league-baseball-public/odds_data/df_odds_history_hits.pkl")
    return merge_prediction_odds(df_prediction, df_odds)

def get_df_history_prediction_odds_2hits_odds():
    df_prediction = read_df_history_prediction_from_gcs(GCS_URL_HISTORY_PREDICTION_2HITS)
    df_odds = read_df_odds_from_gcs("https://storage.googleapis.com/major-league-baseball-public/odds_data/df_odds_history_hits.pkl")
    return merge_prediction_odds(df_prediction, df_odds)

def get_df_history_prediction_odds_1strikeouts_odds():
    df_prediction = read_df_history_prediction_from_gcs(GCS_URL_HISTORY_PREDICTION_1STRIKEOUT)
    df_odds = read_df_odds_from_gcs("https://storage.googleapis.com/major-league-baseball-public/odds_data/df_odds_history_strikeouts.pkl")
    return merge_prediction_odds(df_prediction, df_odds)

@app.callback(
    Output(component_id='confident_over_bet_profit_1hits', component_property='children'),
    Input(component_id='threshold', component_property='value')
)
def update_confident_1hits_over_bet_profit(threshold):
    return get_confident_bets_description(get_df_history_prediction_odds_1hits_odds(), target_prediction_label=1.0, over_or_under="over", target_line=0.5, score_threshold=threshold)

@app.callback(
    Output(component_id='confident_under_bet_profit_1hits', component_property='children'),
    Input(component_id='threshold', component_property='value')
)
def update_confident_1hits_under_bet_profit(threshold):
    return get_confident_bets_description(get_df_history_prediction_odds_1hits_odds(), target_prediction_label=0.0, over_or_under="under", target_line=0.5, score_threshold=threshold)

@app.callback(
    Output(component_id='confident_under_bet_profit_2hits', component_property='children'),
    Input(component_id='threshold', component_property='value')
)
def update_confident_2hits_under_bet_profit(threshold):
    return get_confident_bets_description(get_df_history_prediction_odds_2hits_odds(), target_prediction_label=0.0, over_or_under="under", target_line=1.5, score_threshold=threshold)

@app.callback(
    Output(component_id='confident_over_bet_profit_1strikeouts', component_property='children'),
    Input(component_id='threshold', component_property='value')
)
def update_confident_1strikeouts_over_bet_profit(threshold):
    return get_confident_bets_description(get_df_history_prediction_odds_1strikeouts_odds(), target_prediction_label=1.0, over_or_under="over", target_line=0.5, score_threshold=threshold)

@app.callback(
    Output(component_id='confident_under_bet_profit_1strikeouts', component_property='children'),
    Input(component_id='threshold', component_property='value')
)
def update_confident_1strikeouts_under_bet_profit(threshold):
    return get_confident_bets_description(get_df_history_prediction_odds_1strikeouts_odds(), target_prediction_label=0.0, over_or_under="under", target_line=0.5, score_threshold=threshold)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

