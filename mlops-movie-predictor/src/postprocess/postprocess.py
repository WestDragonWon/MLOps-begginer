import os

from sqlalchemy import create_engine, text
import pandas as pd


def get_engine(db_name):
    engine = create_engine(url=(
        f"mysql+mysqldb://"
        f"{os.environ.get('DB_USER')}:"
        f"{os.environ.get('DB_PASSWORD')}@"
        f"{os.environ.get('DB_HOST')}:"
        f"{os.environ.get('DB_PORT')}/"
        f"{db_name}"))
    return engine


def write_db(data: pd.DataFrame, db_name, table_name):
    engine = get_engine(db_name)
    connect = engine.connect()
    data.to_sql(table_name, connect, if_exists="append")
    connect.close()


def read_db(db_name, table_name, k=10):
    engine = get_engine(db_name)
    connect = engine.connect()
    result = connect.execute(
        statement=text(
            f"select recommend_content_id from {table_name} order by `index` desc limit :k"
        ),
        parameters={"table_name": table_name, "k": k}
    )
    connect.close()
    contents = [data[0] for data in result]
    return contents

