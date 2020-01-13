# %%
import pymongo
import dns
from environs import Env

def get_db_connection():
    env = Env()
    db_url = env('DATABASE_URL')
    dbClient = pymongo.MongoClient(db_url)

    return dbClient

# mydb = dbClient["stocksdb"]
# mycol = mydb["customers"]
# mydict = { "name": "John", "address": "Highway 37" }
# x = mycol.insert_one(mydict)

# %%
