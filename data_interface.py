import json
import requests
import datetime
import time
from sqlalchemy import create_engine
import psycopg2
import pandas as pd


class PostgreLink:

    def __init__(self, database, host='47.114.150.122', port='8888', username='postgres', password='dttest'):
        self.database = database
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.conn = f'postgresql+psycopg2://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}'

    def select_to_df(self, table, fields="*", cond="", print_query=False):
        engine = create_engine(self.conn)
        conn = engine.connect()
        query = f"SELECT {fields} from {table} {cond}"
        if print_query:
            print(query)
        data = pd.read_sql(query, con=conn, chunksize=1000)
        try:
            df = pd.concat(data, ignore_index=True)
        except:
            df = pd.DataFrame()
        conn.close()
        return df

    def insert_many(self, table, df, if_exists='append'):
        engine = create_engine(self.conn)
        conn = engine.connect()
        df.to_sql(table, con=conn, index=False, if_exists=if_exists, chunksize=1000, method='multi')
        conn.close()

    def clear_table(self, table):
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.username,
            password=self.password,
            database=self.database
        )
        cursor = conn.cursor()
        cursor.execute(f"TRUNCATE TABLE {table}")
        conn.commit()
        cursor.close()
        conn.close()

    def commit_query(self, query):
        conn = psycopg2.connect(
            host=self.host,
            port=self.port,
            user=self.username,
            password=self.password,
            database=self.database
        )
        cursor = conn.cursor()
        cursor.execute(query)
        conn.commit()
        cursor.close()
        conn.close()


def ts_query(point_id, from_time, to_time, interval=1, type_=3, print_post=False):
    url = "http://192.168.3.54:8081/data/getpointdata"
    post_data = {
        "point_id": point_id,
        "begin": round(from_time.timestamp()),
        "end": round(to_time.timestamp()),
        "interval": interval,
        "type": type_,
    }
    if print_post:
        print(post_data)
    max_try = 10
    while max_try > 0:
        try:
            resp = requests.post(url=url, data=json.dumps(post_data))
            ts_data = resp.json()['data']
            if ts_data:
                max_try = -1
                res = pd.DataFrame(ts_data)
                res.set_index(['ts'], inplace=True)
                return res
            else:
                max_try -= 1
        except Exception as e:
            # print(e.args)
            # print(str(e))
            print(repr(e))
            max_try -= 1
            time.sleep(1)
    if max_try == 0:
        print("\nquery failed")


def ts_querys(point_ids, from_time, to_time, interval=1, type_=3, print_post=False) -> dict:
    url = "http://192.168.3.54:8081/data/getpointsdata"
    post_data = {
        "point_ids": point_ids,
        "begin": round(from_time.timestamp()),
        "end": round(to_time.timestamp()),
        "interval": interval,
        "type": type_,
    }
    if print_post:
        print(post_data)
    max_try = 10
    while max_try > 0:
        try:
            resp = requests.post(url=url, data=json.dumps(post_data))
            data = resp.json()['data']
            if data:
                max_try = -1
                res = dict()
                for point in data:
                    res[point['point_id']] = pd.DataFrame(point['data'])
                    res[point['point_id']].set_index(['ts'], inplace=True)
                return res
            else:
                max_try -= 1
                time.sleep(1)
        except Exception as e:
            # print(e.args)
            # print(str(e))
            print(repr(e))
            max_try -= 1
            time.sleep(1)
    if max_try == 0:
        print("\nquery failed")


def sum_ts_querys(point_ids, from_time, to_time, interval=1, type_=3, print_post=False):
    return sum(_ for _ in ts_querys(point_ids, from_time, to_time, interval, type_, print_post).values())


def query_data_fc(ts, eq_id, model="ArimaPretrain", max_try=100, sleep=60):
    db = PostgreLink("timeseries")
    table_his = "test_his"
    for _ in range(max_try):
        df_res = db.select_to_df(
            table_his,
            fields="*",
            cond=(f"WHERE model = {repr(model)}" +
                  f" AND ts = '{ts}'" +
                  f" AND eq_id = {repr(eq_id)}")
        )
        if len(df_res):
            return df_res
        else:
            time.sleep(sleep)
    print(f"\nquery data fc failed {max_try}, max try arrived!!!")
    return None


def sv_fc(fc_value, ts: datetime.datetime, eq_id, model="model", interval=1):
    interval_timed = datetime.timedelta(minutes=interval)
    db = PostgreLink("timeseries")
    table_his = "test_his"
    # table_ol = "test_ol"
    df_res = pd.DataFrame(
        {
            'ts': [ts] * len(fc_value),
            'fc_ts': [ts + (i+1) * interval_timed for i in range(len(fc_value))],
            'eq_id': [eq_id] * len(fc_value),
            'value': list(fc_value),
            'model': [model] * len(fc_value)
        }
    )
    # db.clear_table(table_ol)
    # db.insert_many(table_ol, df_res)
    db.insert_many(table_his, df_res)
    print(f"\n{model} for {eq_id} saved at {ts}")


def sv_tsp(ts: datetime.datetime, eq_id, ac1, ac2, ac3, ac4, ac5):
    db = PostgreLink("timeseries")
    table_his = "test_tsp"
    df_res = pd.DataFrame(
        {
            'ts': [ts],
            'eq_id': [eq_id],
            'ac1': [ac1],
            'ac2': [ac2],
            'ac3': [ac3],
            'ac4': [ac4],
            'ac5': [ac5],
        }
    )
    db.insert_many(table_his, df_res)
    print(f"\ntsp for {eq_id} saved at {ts}")


def eval_save(model, eq_id, train_ts, r2, mae, mse, rmse):
    db = PostgreLink("timeseries")
    table_model = "model_test"
    df_res = pd.DataFrame({
        "model": [model],
        "eq_id": [eq_id],
        "train_ts": [train_ts],
        "r2": [r2],
        "mae": [mae],
        "mse": [mse],
        "rmse": [rmse],
    })
    db.insert_many(table_model, df_res)

