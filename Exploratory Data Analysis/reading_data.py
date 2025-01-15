# Imports
import sqlite3 as sq3
import pandas.io.sql as pds
import pandas as pd

# file path
data = 'data/classic_rock.db'

# establish connection
con = sq3.Connection(data)

# Write query
query = '''
SELECT Artist, Release_Year, COUNT(*) AS num_songs, AVG(PlayCount) AS avg_plays  
    FROM rock_songs
    GROUP BY Artist, Release_Year
    ORDER BY num_songs desc;
'''

# execute query
observation = pds.read_sql(query, con)

print(observation.head())

# extra parameters include:
#   coerce_float: Attempt to force numbers into floats
#   parse_dates: List of columns to parse as dates
#   chunksize: Number of rows to include in each chunk

path = 'data/baseball.db'
con = sq3.Connection(path)
query = '''
SELECT *
    FROM allstarfull;
'''

observation = pds.read_sql(query, con)

print(observation.head())