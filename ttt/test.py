import json
from dataclasses import dataclass

from dotenv import load_dotenv

from hyperdb import HyperDB

load_dotenv()

# Load documents from the JSONL file
documents = []

# with open("demo/pokemon.jsonl", "r") as f:
#     for line in f:
#         documents.append(json.loads(line))
#
# # Instantiate HyperDB with the list of documents
# db = HyperDB(documents, key="info.description")
#
# # Save the HyperDB instance to a file
# db.save("demo/pokemon_hyperdb.pickle.gz")
#
# # Load the HyperDB instance from the save file
# db.load("demo/pokemon_hyperdb.pickle.gz")
#
# # Query the HyperDB instance with a text input
# results = db.query("Likes to sleep.", top_k=5)


@dataclass
class DocString:
    name: str
    age: int
    city: str
    occupation: str

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        _str = ""
        for key, value in self.__dict__.items():
            _str += f"{key}: {value}\n"
        return _str

    def to_dict(self):
        # Add the string representation of the document to the dictionary
        self.__dict__["full"] = self.__str__()
        return self.__dict__


john = DocString("John", 30, "New York", "Engineer")
jane = DocString("Jane", 35, "London", "Engineer")
joe = DocString("Joe", 40, "New York", "Engineer")

docs = [john, jane, joe]
docs = [doc.to_dict() for doc in docs]

db = HyperDB(docs, key="full")
