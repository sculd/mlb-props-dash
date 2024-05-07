# Import packages
from dash import Dash, dash_table, dcc, html, Input, Output, ctx
import pandas as pd, numpy as np
import datetime, pytz

from flask_caching import Cache

date_str_today = datetime.datetime.now(pytz.timezone('US/Pacific')).strftime("%Y-%m-%d")
date_str_yesterday = (datetime.datetime.now(pytz.timezone('US/Pacific')) - datetime.timedelta(days=1)).strftime("%Y-%m-%d")

_default_threshold = 0.75
_live_data_table_page_size = 6
_history_data_table_page_size = 7

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
    if len(df) > 0:
        df = df[df.game_date <= date_str_yesterday].sort_values(['game_date'], ascending=False)

    return df

def get_confident_bets_description(desc, df_prediction_odds, target_prediction_label, over_or_under, target_line, score_threshold=0.75):
    df_confident_prediction_odds = df_prediction_odds[
        df_prediction_odds["prediction_score"] >= score_threshold]

    # the prediction_label should be separatedly checked. higher score does not always lead to prediction label. (the score stands for both labels).
    df_confident_prediction_odds = df_confident_prediction_odds[df_confident_prediction_odds["prediction_label"] == target_prediction_label]
    if target_line is None:
        df_confident_prediction_odds_opposite_line = df_confident_prediction_odds
        df_confident_prediction_odds = df_confident_prediction_odds
    else:
        if over_or_under == "over":
            df_confident_prediction_odds_opposite_line = df_confident_prediction_odds[df_confident_prediction_odds.over_line != target_line]
            df_confident_prediction_odds = df_confident_prediction_odds[df_confident_prediction_odds.over_line == target_line]
        else:
            df_confident_prediction_odds_opposite_line = df_confident_prediction_odds[df_confident_prediction_odds.under_line != target_line]
            df_confident_prediction_odds = df_confident_prediction_odds[df_confident_prediction_odds.under_line == target_line]

    if len(df_confident_prediction_odds) == 0:
        win_rate = 0.0
    else:
        win_rate = 1.0 * len(df_confident_prediction_odds[df_confident_prediction_odds.property_value == df_confident_prediction_odds.prediction_label]) / len(df_confident_prediction_odds)
        win_rate = round(win_rate, 3)

    if len(df_confident_prediction_odds) == 0:
        return f"{desc} success recorded ratio: {win_rate}"

    if over_or_under == "over":
        ideal_profit_neg_odds = np.divide(100.0, np.abs(df_confident_prediction_odds.over_odds))
        ideal_profit_pos_odds = np.divide(df_confident_prediction_odds.over_odds, 100)
        ideal_profit = np.where(df_confident_prediction_odds.over_odds < 0, ideal_profit_neg_odds, ideal_profit_pos_odds)

        ideal_profit_reverse_neg_odds = np.divide(100.0, np.abs(df_confident_prediction_odds.under_odds))
        ideal_profit_reverse_pos_odds = np.divide(df_confident_prediction_odds.under_odds, 100)
        ideal_profit_reverse = np.where(df_confident_prediction_odds.under_odds < 0, ideal_profit_reverse_neg_odds, ideal_profit_reverse_pos_odds)
    else:
        ideal_profit_neg_odds = np.divide(100.0, np.abs(df_confident_prediction_odds.under_odds))
        ideal_profit_pos_odds = np.divide(df_confident_prediction_odds.under_odds, 100)
        ideal_profit = np.where(df_confident_prediction_odds.under_odds < 0, ideal_profit_neg_odds, ideal_profit_pos_odds)

        ideal_profit_reverse_neg_odds = np.divide(100.0, np.abs(df_confident_prediction_odds.over_odds))
        ideal_profit_reverse_pos_odds = np.divide(df_confident_prediction_odds.over_odds, 100)
        ideal_profit_reverse = np.where(df_confident_prediction_odds.over_odds < 0, ideal_profit_reverse_neg_odds, ideal_profit_reverse_pos_odds)

    l = len(df_confident_prediction_odds)

    ideal_reward = np.add(1.0, ideal_profit)
    prodiction_successes = len(df_confident_prediction_odds[df_confident_prediction_odds.property_value == target_prediction_label])
    profit = np.sum(np.multiply(np.where(df_confident_prediction_odds.property_value == target_prediction_label, 1, 0), ideal_reward)) - l
    profit = round(profit, 2)

    # profit in case the choice was made to the opposite of the model prediction.
    prodiction_reverse_successes = len(df_confident_prediction_odds[df_confident_prediction_odds.property_value != target_prediction_label])
    ideal_reward_reverse = np.add(1.0, ideal_profit_reverse)
    profit_reverse = np.sum(np.multiply(np.where(df_confident_prediction_odds.property_value != target_prediction_label, 1, 0), ideal_reward_reverse)) - l
    profit_reverse = round(profit_reverse, 2)
    success_ratio = round(1.0 * prodiction_successes / l, 3) if l > 0 else 0
    return f'{desc} excluded ({len(df_confident_prediction_odds_opposite_line)}) which is of different line than {target_line}, success recorded ratio: {success_ratio} ({prodiction_successes} out of {l}), profit: {profit}, profit_reverse: {profit_reverse}'

def get_positive_under_odds_bets_description(desc, df_prediction_odds, target_line, positive_odds_threshold):
    '''
    This describes the performance when betting on the under (property value 0), positive odds, regardless of the prediction.
    '''
    df_confident_prediction_odds = df_prediction_odds.dropna()
    df_confident_prediction_odds = df_confident_prediction_odds[
        df_confident_prediction_odds["game_date"] >= '2023-07-19']
    df_confident_prediction_odds = df_confident_prediction_odds[
        df_confident_prediction_odds["under_odds"] >= positive_odds_threshold]
    df_confident_prediction_odds = df_confident_prediction_odds[df_confident_prediction_odds.under_line == target_line]
    if len(df_confident_prediction_odds) == 0:
        return f"{desc} empty"

    ideal_profit = np.divide(df_confident_prediction_odds.under_odds, 100)
    ideal_reward = np.add(1.0, ideal_profit)
    prodiction_successes = len(df_confident_prediction_odds[df_confident_prediction_odds.property_value == 0])
    l = len(df_confident_prediction_odds)
    profit = np.sum(np.multiply(np.where(df_confident_prediction_odds.property_value == 0, 1, 0), ideal_reward)) - l
    profit = round(profit, 2)

    success_ratio = round(1.0 * prodiction_successes / l, 3) if l > 0 else 0
    return f'{desc} success recorded ratio: {success_ratio} ({prodiction_successes} out of {l}), profit: {profit}'

# App layout
app.layout = html.Div([
    html.Div(
        [
            "Threshold",
            dcc.Input(id="threshold", type="number", value=_default_threshold, step=0.025),
            "Blind Pick Pos Odds",
            dcc.Input(id="pos_odds_thresholdd", type="number", value=150, step=10),
            "Keep null records",
            dcc.Checklist(
                ['null'],
                ['null'],
                id="keep_null"
            ),
            dcc.Checklist(
                ['all_lines'],
                [],
                id="all_lines"
            ),
            dcc.Dropdown(['all', 'win', 'lose'], 'all', id='win_loss_dropdown'),
        ],
        style={"width": 250},
    ),
    dcc.Tabs(id='property-tabs', value='2Hits', children=[
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
            dash_table.DataTable(id="live_table_1hits",
                                 data=get_live_data(GCS_URL_LIVE_PREDICTION_1HITS, GCS_URL_LIVE_ODDS_HITS, _default_threshold, keep_null=True, target_line=None),
                                 columns=[
                                    {"name": i, 'id': i} for i in get_live_df(GCS_URL_LIVE_PREDICTION_1HITS, GCS_URL_LIVE_ODDS_HITS, _default_threshold, keep_null=True, target_line=None).columns
                                 ],
                                 filter_action="native",
                                 page_size=_live_data_table_page_size),
            html.Div(children='Prediction History'),
            dash_table.DataTable(id="history_table_1hits",
                                 data=get_history_data(GCS_URL_HISTORY_PREDICTION_1HITS, GCS_URL_HISTORY_ODDS_HITS, _default_threshold, keep_null=True, target_line=None, win_loss_dropdown='all'),
                                 columns=[
                                    {"name": i, 'id': i} for i in get_history_df(GCS_URL_HISTORY_PREDICTION_1HITS, GCS_URL_HISTORY_ODDS_HITS, _default_threshold, keep_null=True, target_line=None, win_loss_dropdown='all').columns
                                 ],
                                 filter_action="native",
                                 page_size=_history_data_table_page_size),
            html.Div(id='confident_over_bet_profit_1hits'),
            html.Div(id='confident_under_bet_profit_1hits'),
            html.Div(id='blind_under_bet_profit_1hits'),
        ])
    elif tab == '2Hits':
        return html.Div([
            html.Div(children='Live Prediction'),
            dash_table.DataTable(id="live_table_2hits",
                                 data=get_live_data(GCS_URL_LIVE_PREDICTION_2HITS, GCS_URL_LIVE_ODDS_HITS, _default_threshold, keep_null=True, target_line=None),
                                 columns=[
                                    {"name": i, 'id': i} for i in get_live_df(GCS_URL_LIVE_PREDICTION_2HITS, GCS_URL_LIVE_ODDS_HITS, _default_threshold, keep_null=True, target_line=None).columns
                                 ],
                                 filter_action="native",
                                 page_size=_live_data_table_page_size),
            html.Div(children='Prediction History'),
            dash_table.DataTable(id="history_table_2hits",
                                 data=get_history_data(GCS_URL_HISTORY_PREDICTION_2HITS, GCS_URL_HISTORY_ODDS_HITS, _default_threshold, keep_null=True, target_line=None, win_loss_dropdown='all'),
                                 columns=[
                                    {"name": i, 'id': i} for i in get_history_df(GCS_URL_HISTORY_PREDICTION_2HITS, GCS_URL_HISTORY_ODDS_HITS, _default_threshold, keep_null=True, target_line=None, win_loss_dropdown='all').columns
                                 ],
                                 filter_action="native",
                                 page_size=_history_data_table_page_size),
            html.Div(id='confident_under_bet_profit_2hits'),
        ])
    elif tab == '1Strikeouts':
        return html.Div([
            html.Div(children='Live Prediction'),
            dash_table.DataTable(id="live_table_1strikeouts",
                                 data=get_live_data(GCS_URL_LIVE_PREDICTION_1STRIKEOUT, GCS_URL_LIVE_ODDS_STRIKEOUTS, _default_threshold, keep_null=True, target_line=None),
                                 columns=[
                                    {"name": i, 'id': i} for i in get_live_df(GCS_URL_LIVE_PREDICTION_1STRIKEOUT, GCS_URL_LIVE_ODDS_STRIKEOUTS, _default_threshold, keep_null=True, target_line=None).columns
                                 ],
                                 filter_action="native",
                                 page_size=_live_data_table_page_size),
            html.Div(children='Prediction History'),
            dash_table.DataTable(id="history_table_1strikeouts",
                                 data=get_history_data(GCS_URL_HISTORY_PREDICTION_1STRIKEOUT, GCS_URL_HISTORY_ODDS_STRIKEOUTS, _default_threshold, keep_null=True, target_line=None, win_loss_dropdown='all'),
                                 columns=[
                                    {"name": i, 'id': i} for i in get_history_df(GCS_URL_HISTORY_PREDICTION_1STRIKEOUT, GCS_URL_HISTORY_ODDS_STRIKEOUTS, _default_threshold, keep_null=True, target_line=None, win_loss_dropdown='all').columns
                                 ],
                                 filter_action="native",
                                 page_size=_history_data_table_page_size),
            html.Div(id='confident_over_bet_profit_1strikeouts'),
            html.Div(id='confident_under_bet_profit_1strikeouts'),
        ])

def merge_prediction_odds(df_prediction, df_odds, keep_null):
    columns_odds = ["game_id", "batting_name"] + list(df_odds.columns.difference(df_prediction.columns))
    df_odds_selected = df_odds[columns_odds]
    join_how = 'left' if keep_null else 'inner'
    return df_prediction.merge(df_odds_selected, on=["game_id", "batting_name"], how=join_how)

def filter_df(df_prediction, df_odds, threshold, keep_null, target_line, win_loss_dropdown):
    df_prediction_odds = merge_prediction_odds(df_prediction, df_odds, keep_null)
    df_prediction_odds_high_score = df_prediction_odds[(df_prediction_odds.prediction_score > threshold)]
    if not keep_null:
        df_prediction_odds_high_score = df_prediction_odds_high_score.dropna()
    if target_line is not None:
        df_prediction_odds_high_score = df_prediction_odds_high_score[df_prediction_odds_high_score.over_line == target_line]
    if win_loss_dropdown == 'win':
        df_prediction_odds_high_score = df_prediction_odds_high_score[df_prediction_odds_high_score.prediction_label == df_prediction_odds_high_score.property_value]
    elif win_loss_dropdown == 'lose':
        df_prediction_odds_high_score = df_prediction_odds_high_score[df_prediction_odds_high_score.prediction_label != df_prediction_odds_high_score.property_value]
    return df_prediction_odds_high_score

def get_live_df(live_prediction_gcs, live_odds_gcs, threshold, keep_null, target_line):
    df_prediction = read_df_live_prediction_from_gcs(live_prediction_gcs)
    df_odds = read_df_odds_from_gcs(live_odds_gcs)

    return filter_df(df_prediction, df_odds, threshold, keep_null, target_line, win_loss_dropdown='all')

def get_live_data(live_prediction_gcs, live_odds_gcs, threshold, keep_null, target_line):
    return get_live_df(live_prediction_gcs, live_odds_gcs, threshold, keep_null, target_line).to_dict("records")

def get_history_df(history_prediction_gcs, history_odds_gcs, threshold, keep_null, target_line, win_loss_dropdown):
    df_prediction = read_df_history_prediction_from_gcs(history_prediction_gcs)
    df_odds = read_df_odds_from_gcs(history_odds_gcs)

    return filter_df(df_prediction, df_odds, threshold, keep_null, target_line, win_loss_dropdown)

def get_history_data(history_prediction_gcs, history_odds_gcs, threshold, keep_null, target_line, win_loss_dropdown):
    return get_history_df(history_prediction_gcs, history_odds_gcs, threshold, keep_null, target_line, win_loss_dropdown).to_dict("records")

@app.callback(
    Output("live_table_1hits", "data"), Input("threshold", "value"), Input("keep_null", "value"), Input("all_lines", "value")
)
def update_live_table_1hits(threshold, keep_null, all_lines):
    return get_live_data(
        GCS_URL_LIVE_PREDICTION_1HITS,
        GCS_URL_LIVE_ODDS_HITS,
        threshold, 'null' in keep_null, None if 'all_lines' in all_lines else 0.5)

@app.callback(
    Output("live_table_2hits", "data"), Input("threshold", "value"), Input("keep_null", "value"), Input("all_lines", "value")
)
def update_live_table_2hits(threshold, keep_null, all_lines):
    # target line is None as 2hits below corresponds to both 0.5 and 1.5
    # negative 2hits means 0 or 1 hit so 1.5 line under
    return get_live_data(
        GCS_URL_LIVE_PREDICTION_2HITS,
        GCS_URL_LIVE_ODDS_HITS,
        threshold, 'null' in keep_null, None if 'all_lines' in all_lines else 1.5)

@app.callback(
    Output("live_table_1strikeouts", "data"), Input("threshold", "value"), Input("keep_null", "value"), Input("all_lines", "value")
)
def update_live_table_1strikeout(threshold, keep_null, all_lines):
    return get_live_data(
        GCS_URL_LIVE_PREDICTION_1STRIKEOUT,
        GCS_URL_LIVE_ODDS_STRIKEOUTS,
        threshold, 'null' in keep_null, None if 'all_lines' in all_lines else 0.5)

@app.callback(
    Output("history_table_1hits", "data"), Input("threshold", "value"), Input("keep_null", "value"), Input("all_lines", "value"), Input("win_loss_dropdown", "value")
)
def update_history_table_1hits(threshold, keep_null, all_lines, win_loss_dropdown):
    return get_history_data(
        GCS_URL_HISTORY_PREDICTION_1HITS,
        GCS_URL_HISTORY_ODDS_HITS,
        threshold, 'null' in keep_null, None if 'all_lines' in all_lines else 0.5, win_loss_dropdown)

@app.callback(
    Output("history_table_2hits", "data"), Input("threshold", "value"), Input("keep_null", "value"), Input("all_lines", "value"), Input("win_loss_dropdown", "value")
)
def update_history_table_2hits(threshold, keep_null, all_lines, win_loss_dropdown):
    return get_history_data(
        GCS_URL_HISTORY_PREDICTION_2HITS,
        GCS_URL_HISTORY_ODDS_HITS,
        threshold, 'null' in keep_null, None if 'all_lines' in all_lines else 1.5, win_loss_dropdown)

@app.callback(
    Output("history_table_1strikeouts", "data"), Input("threshold", "value"), Input("keep_null", "value"), Input("all_lines", "value"), Input("win_loss_dropdown", "value")
)
def update_history_table_1strikeout(threshold, keep_null, all_lines, win_loss_dropdown):
    return get_history_data(
        GCS_URL_HISTORY_PREDICTION_1STRIKEOUT,
        GCS_URL_HISTORY_ODDS_STRIKEOUTS,
        threshold, 'null' in keep_null, None if 'all_lines' in all_lines else 0.5, win_loss_dropdown)

def get_df_history_prediction_odds_1hits_odds(threshold, keep_null, all_lines, win_loss_dropdown):
    df_prediction = read_df_history_prediction_from_gcs(GCS_URL_HISTORY_PREDICTION_1HITS)
    df_odds = read_df_odds_from_gcs(GCS_URL_HISTORY_ODDS_HITS)
    return filter_df(df_prediction, df_odds,
              threshold, 'null' in keep_null, None if 'all_lines' in all_lines else 0.5,
              win_loss_dropdown)

def get_df_history_prediction_odds_2hits_odds(threshold, keep_null, all_lines, win_loss_dropdown):
    df_prediction = read_df_history_prediction_from_gcs(GCS_URL_HISTORY_PREDICTION_2HITS)
    df_odds = read_df_odds_from_gcs(GCS_URL_HISTORY_ODDS_HITS)
    return filter_df(df_prediction, df_odds,
              threshold, 'null' in keep_null, None if 'all_lines' in all_lines else 1.5,
              win_loss_dropdown)

def get_df_history_prediction_odds_1strikeouts_odds(threshold, keep_null, all_lines, win_loss_dropdown):
    df_prediction = read_df_history_prediction_from_gcs(GCS_URL_HISTORY_PREDICTION_1STRIKEOUT)
    df_odds = read_df_odds_from_gcs(GCS_URL_HISTORY_ODDS_STRIKEOUTS)
    return filter_df(df_prediction, df_odds,
              threshold, 'null' in keep_null, None if 'all_lines' in all_lines else 0.5,
              win_loss_dropdown)

@app.callback(
    Output(component_id='confident_over_bet_profit_1hits', component_property='children'),
    Input("threshold", "value"), Input("keep_null", "value"), Input("all_lines", "value"), Input("win_loss_dropdown", "value")
)
def update_confident_1hits_over_bet_profit(threshold, keep_null, all_lines, win_loss_dropdown):
    return get_confident_bets_description(
        "1hit line=0.5 over",
        get_df_history_prediction_odds_1hits_odds(threshold, keep_null, all_lines, win_loss_dropdown),
        target_prediction_label=1.0, over_or_under="over",
        target_line=None if 'all_lines' in all_lines else 0.5, score_threshold=threshold)

@app.callback(
    Output(component_id='confident_under_bet_profit_1hits', component_property='children'),
    Input(component_id='threshold', component_property='value'), Input("keep_null", "value"), Input("all_lines", "value"), Input("win_loss_dropdown", "value")
)
def update_confident_1hits_under_bet_profit(threshold, keep_null, all_lines, win_loss_dropdown):
    return get_confident_bets_description(

        "1hit line=0.5 under",
        get_df_history_prediction_odds_1hits_odds(threshold, keep_null, all_lines, win_loss_dropdown),
        target_prediction_label=0.0, over_or_under="under",
        target_line=None if 'all_lines' in all_lines else 0.5, score_threshold=threshold)

@app.callback(
    Output(component_id='blind_under_bet_profit_1hits', component_property='children'),
    Input(component_id='pos_odds_thresholdd', component_property='value'), Input("keep_null", "value"), Input("all_lines", "value"), Input("win_loss_dropdown", "value")
)
def update_blins_1hits_under_bet_profit(pos_odds_thresholdd, keep_null, all_lines, win_loss_dropdown):
    return get_positive_under_odds_bets_description(
        f"1hit line=0.5 bet all positive under odds > {pos_odds_thresholdd}",
        get_df_history_prediction_odds_1hits_odds(pos_odds_thresholdd, keep_null, all_lines, win_loss_dropdown),
        target_line=None if 'all_lines' in all_lines else 0.5, positive_odds_threshold=pos_odds_thresholdd)

@app.callback(
    Output(component_id='confident_under_bet_profit_2hits', component_property='children'),
    Input(component_id='threshold', component_property='value'), Input("keep_null", "value"), Input("all_lines", "value"), Input("win_loss_dropdown", "value")
)
def update_confident_2hits_under_bet_profit(threshold, keep_null, all_lines, win_loss_dropdown):
    return get_confident_bets_description(
        "2hits line=1.5 under",
        get_df_history_prediction_odds_2hits_odds(threshold, keep_null, all_lines, win_loss_dropdown),
        target_prediction_label=0.0, over_or_under="under",
        target_line=None if 'all_lines' in all_lines else 1.5, score_threshold=threshold)

@app.callback(
    Output(component_id='confident_over_bet_profit_1strikeouts', component_property='children'),
    Input(component_id='threshold', component_property='value'), Input("keep_null", "value"), Input("all_lines", "value"), Input("win_loss_dropdown", "value")
)
def update_confident_1strikeouts_over_bet_profit(threshold, keep_null, all_lines, win_loss_dropdown):
    return get_confident_bets_description(
        "1striekout line=0.5 over",
        get_df_history_prediction_odds_1strikeouts_odds(threshold, keep_null, all_lines, win_loss_dropdown),
        target_prediction_label=1.0, over_or_under="over",
        target_line=None if 'all_lines' in all_lines else 0.5, score_threshold=threshold)

@app.callback(
    Output(component_id='confident_under_bet_profit_1strikeouts', component_property='children'),
    Input(component_id='threshold', component_property='value'), Input("keep_null", "value"), Input("all_lines", "value"), Input("win_loss_dropdown", "value")
)
def update_confident_1strikeouts_under_bet_profit(threshold, keep_null, all_lines, win_loss_dropdown):
    return get_confident_bets_description(
        "1striekout line=0.5 under",
        get_df_history_prediction_odds_1strikeouts_odds(threshold, keep_null, all_lines, win_loss_dropdown),
        target_prediction_label=0.0, over_or_under="under",
        target_line=None if 'all_lines' in all_lines else 0.5, score_threshold=threshold)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

