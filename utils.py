import sqlalchemy
import numpy as np
import pandas as pd
from pandas.io import sql

def filters(loans, rules_file):
    with open(rules_file) as f:
        loans = Dfexpr(loans, f, [])
    loans = loans[np.logical_not(loans.filtered)]
    return loans
                
def Dfexpr(df, expressions, categorical_vars=[]):
    for exp in expressions:
        df.eval(exp, inplace=True)
    for cat in categorical_vars:
        df = pd.concat([df, pd.get_dummies(df[cat], prefix=cat, prefix_sep="")], axis=1)

    return df

def myloans(engine):
    """
    get all notes I own
    """
    existingloans = pd.read_sql('select distinct "id" from lc_id', engine)
    return existingloans.id.values


def add_column(engine, table_name, column_name, column_type):
    engine.execute('ALTER TABLE %s ADD COLUMN %s %s' % (table_name, column_name, column_type))

def append_db(df, table_name, engine):
    """
    append data from db to database, add new columns from db to table if necessary
    """
    metadata = sqlalchemy.MetaData()
    try:
        tbl = sqlalchemy.Table(table_name, metadata, autoload=True, autoload_with=engine)
        existing_cols = [c.name for c in tbl.columns]
        for newcol in set(df.columns) - set(existing_cols):
            #pdb.set_trace()
            db = sql.SQLiteDatabase(engine, 'sqlite')
            table = sql.SQLiteTable(table_name, db, frame=df, index=False,
                                if_exists="append")
            #hacky, called protected function to get sql typename from pandas dtype
            add_column(engine, table_name, newcol, table._sql_type_name(df[newcol]))
    except sqlalchemy.exc.NoSuchTableError:
        pass
    df.to_sql(table_name, engine, if_exists='append', index=False, chunksize=1000)
