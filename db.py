from tinydb import TinyDB, Query
import datetime

class DataBase:
    #database will have 1) ID, 2) Type(input or output), 3) Image Directory(name) on the server.
    def __init__(self,dbName):
        self.db = TinyDB(dbName)

        pass

    def insertImage(self,type,dir):
        self.db.insert({'id':str(datetime.datetime.now()),'type':type,'dir':dir})
    
    def deleteImage(self):
        self.db

    def __str__(self):
        return str(self.db)

if __name__ == "__main__":
    db=TinyDB("db.json")
