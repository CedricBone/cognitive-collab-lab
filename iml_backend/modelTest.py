import json
import pickle
from model import NERModel  # Import your NER model class from model.py




def pretty_print_result(result):
   # Score and cluster (if any)
   score = result.get("score", "No score")
   cluster = result.get("cluster", "No cluster")
   print(f"  Score: {score}, Cluster: {cluster}")


   # Individual entities
   for entity in result["result"]:
       from_name = entity["from_name"]
       to_name = entity["to_name"]
       label_type = entity["type"]
       label = entity["value"]["labels"][0]
       start = entity["value"]["start"]
       end = entity["value"]["end"]
       text = entity["value"]["text"]


       print(f"    Entity:")
       print(f"      From: {from_name}, To: {to_name}, Type: {label_type}")
       print(f"      Label: {label}, Start: {start}, End: {end}")
       print(f"      Text: {text}")
   print("\n")




tasks = [
   {
       "id": 51,
       "data": {"text": "The quick brown fox jumps over the lazy dog."},
       "meta": {},
       "created_at": "2024-01-11T20:21:52.446292Z",
       "updated_at": "2024-01-11T20:23:23.314934Z",
       "is_labeled": True,
       "overlap": 1,
       "inner_id": 1,
       "total_annotations": 1,
       "cancelled_annotations": 0,
       "total_predictions": 0,
       "comment_count": 0,
       "unresolved_comment_count": 0,
       "last_comment_updated_at": None,
       "project": 2,
       "updated_by": 1,
       "file_upload": 13,
       "comment_authors": [],
       "annotations": [
           {
               "id": 19,
               "created_username": " cb9017@rit.edu, 1",
               "created_ago": "9\xa0hours, 37\xa0minutes",
               "completed_by": 1,
               "result": [
                   {
                       "value": {
                           "start": 35,
                           "end": 39,
                           "text": "lazy",
                           "labels": ["MISC"],
                       },
                       "id": "rQlBjiG6hk",
                       "from_name": "label",
                       "to_name": "text",
                       "type": "labels",
                       "origin": "manual",
                   },
                   {
                       "value": {
                           "start": 4,
                           "end": 9,
                           "text": "quick",
                           "labels": ["MISC"],
                       },
                       "id": "TAvpIhHp28",
                       "from_name": "label",
                       "to_name": "text",
                       "type": "labels",
                       "origin": "manual",
                   },
                   {
                       "value": {
                           "start": 16,
                           "end": 19,
                           "text": "fox",
                           "labels": ["PER"],
                       },
                       "id": "qfjurEWaCK",
                       "from_name": "label",
                       "to_name": "text",
                       "type": "labels",
                       "origin": "manual",
                   },
               ],
               "was_cancelled": False,
               "ground_truth": False,
               "created_at": "2024-01-11T20:23:22.988914Z",
               "updated_at": "2024-01-11T20:23:22.988950Z",
               "draft_created_at": "2024-01-11T20:23:20.268946Z",
               "lead_time": 16.17,
               "import_id": None,
               "last_action": None,
               "task": 51,
               "project": 2,
               "updated_by": 1,
               "parent_prediction": None,
               "parent_annotation": None,
               "last_created_by": None,
           }
       ],
       "predictions": [],
   },
   {
       "id": 52,
       "data": {"text": "Albert Einstein was a physicist born in Germany."},
       "meta": {},
       "created_at": "2024-01-11T20:21:52.446397Z",
       "updated_at": "2024-01-11T20:24:01.556807Z",
       "is_labeled": True,
       "overlap": 1,
       "inner_id": 2,
       "total_annotations": 1,
       "cancelled_annotations": 0,
       "total_predictions": 0,
       "comment_count": 0,
       "unresolved_comment_count": 0,
       "last_comment_updated_at": None,
       "project": 2,
       "updated_by": 1,
       "file_upload": 13,
       "comment_authors": [],
       "annotations": [
           {
               "id": 20,
               "created_username": " cb9017@rit.edu, 1",
               "created_ago": "9\xa0hours, 37\xa0minutes",
               "completed_by": 1,
               "result": [
                   {
                       "value": {
                           "start": 0,
                           "end": 15,
                           "text": "Albert Einstein",
                           "labels": ["PER"],
                       },
                       "id": "mG1CLf5jWO",
                       "from_name": "label",
                       "to_name": "text",
                       "type": "labels",
                       "origin": "manual",
                   },
                   {
                       "value": {
                           "start": 40,
                           "end": 47,
                           "text": "Germany",
                           "labels": ["LOC"],
                       },
                       "id": "RbNZErMvLM",
                       "from_name": "label",
                       "to_name": "text",
                       "type": "labels",
                       "origin": "manual",
                   },
               ],
               "was_cancelled": False,
               "ground_truth": False,
               "created_at": "2024-01-11T20:24:01.523716Z",
               "updated_at": "2024-01-11T20:24:01.523747Z",
               "draft_created_at": "2024-01-11T20:23:59.334105Z",
               "lead_time": 11.368,
               "import_id": None,
               "last_action": None,
               "task": 52,
               "project": 2,
               "updated_by": 1,
               "parent_prediction": None,
               "parent_annotation": None,
               "last_created_by": None,
           }
       ],
       "predictions": [],
   },
   {
       "id": 53,
       "data": {
           "text": "The Great Wall of China is one of the world's most famous landmarks."
       },
       "meta": {},
       "created_at": "2024-01-11T20:21:52.446469Z",
       "updated_at": "2024-01-12T02:14:34.054378Z",
       "is_labeled": True,
       "overlap": 1,
       "inner_id": 3,
       "total_annotations": 1,
       "cancelled_annotations": 0,
       "total_predictions": 0,
       "comment_count": 0,
       "unresolved_comment_count": 0,
       "last_comment_updated_at": None,
       "project": 2,
       "updated_by": 1,
       "file_upload": 13,
       "comment_authors": [],
       "annotations": [
           {
               "id": 21,
               "created_username": " cb9017@rit.edu, 1",
               "created_ago": "3\xa0hours, 46\xa0minutes",
               "completed_by": 1,
               "result": [
                   {
                       "value": {
                           "start": 0,
                           "end": 23,
                           "text": "The Great Wall of China",
                           "labels": ["LOC"],
                       },
                       "id": "6G3Narpb-z",
                       "from_name": "label",
                       "to_name": "text",
                       "type": "labels",
                       "origin": "manual",
                   },
                   {
                       "value": {
                           "start": 38,
                           "end": 45,
                           "text": "world's",
                           "labels": ["LOC"],
                       },
                       "id": "Oix6jesPmF",
                       "from_name": "label",
                       "to_name": "text",
                       "type": "labels",
                       "origin": "manual",
                   },
                   {
                       "value": {
                           "start": 58,
                           "end": 67,
                           "text": "landmarks",
                           "labels": ["MISC"],
                       },
                       "id": "veHb26rVFY",
                       "from_name": "label",
                       "to_name": "text",
                       "type": "labels",
                       "origin": "manual",
                   },
               ],
               "was_cancelled": False,
               "ground_truth": False,
               "created_at": "2024-01-12T02:14:31.731949Z",
               "updated_at": "2024-01-12T02:14:31.731976Z",
               "draft_created_at": "2024-01-12T02:14:24.885255Z",
               "lead_time": 21.355,
               "import_id": None,
               "last_action": None,
               "task": 53,
               "project": 2,
               "updated_by": 1,
               "parent_prediction": None,
               "parent_annotation": None,
               "last_created_by": None,
           }
       ],
       "predictions": [],
   },
   {
       "id": 54,
       "data": {"text": "Shakespeare wrote Romeo and Juliet."},
       "meta": {},
       "created_at": "2024-01-11T20:21:52.446538Z",
       "updated_at": "2024-01-12T05:03:13.882453Z",
       "is_labeled": True,
       "overlap": 1,
       "inner_id": 4,
       "total_annotations": 1,
       "cancelled_annotations": 0,
       "total_predictions": 0,
       "comment_count": 0,
       "unresolved_comment_count": 0,
       "last_comment_updated_at": None,
       "project": 2,
       "updated_by": 1,
       "file_upload": 13,
       "comment_authors": [],
       "annotations": [
           {
               "id": 22,
               "created_username": " cb9017@rit.edu, 1",
               "created_ago": "58\xa0minutes",
               "completed_by": 1,
               "result": [
                   {
                       "value": {
                           "start": 0,
                           "end": 11,
                           "text": "Shakespeare",
                           "labels": ["PER"],
                       },
                       "id": "p67K3c4Sgr",
                       "from_name": "label",
                       "to_name": "text",
                       "type": "labels",
                       "origin": "manual",
                   },
                   {
                       "value": {
                           "start": 18,
                           "end": 23,
                           "text": "Romeo",
                           "labels": ["PER"],
                       },
                       "id": "-keWukHoNd",
                       "from_name": "label",
                       "to_name": "text",
                       "type": "labels",
                       "origin": "manual",
                   },
                   {
                       "value": {
                           "start": 28,
                           "end": 34,
                           "text": "Juliet",
                           "labels": ["PER"],
                       },
                       "id": "MC5agpAh5t",
                       "from_name": "label",
                       "to_name": "text",
                       "type": "labels",
                       "origin": "manual",
                   },
               ],
               "was_cancelled": False,
               "ground_truth": False,
               "created_at": "2024-01-12T05:03:11.510464Z",
               "updated_at": "2024-01-12T05:03:11.510483Z",
               "draft_created_at": "2024-01-12T05:03:09.314913Z",
               "lead_time": 11.656,
               "import_id": None,
               "last_action": None,
               "task": 54,
               "project": 2,
               "updated_by": 1,
               "parent_prediction": None,
               "parent_annotation": None,
               "last_created_by": None,
           }
       ],
       "predictions": [],
   },
   {
       "id": 55,
       "data": {
           "text": "The Amazon Rainforest is the largest tropical rainforest in the world."
       },
       "meta": {},
       "created_at": "2024-01-11T20:21:52.446607Z",
       "updated_at": "2024-01-12T05:41:41.806682Z",
       "is_labeled": True,
       "overlap": 1,
       "inner_id": 5,
       "total_annotations": 1,
       "cancelled_annotations": 0,
       "total_predictions": 0,
       "comment_count": 0,
       "unresolved_comment_count": 0,
       "last_comment_updated_at": None,
       "project": 2,
       "updated_by": 1,
       "file_upload": 13,
       "comment_authors": [],
       "annotations": [
           {
               "id": 23,
               "created_username": " cb9017@rit.edu, 1",
               "created_ago": "36\xa0minutes",
               "completed_by": 1,
               "result": [
                   {
                       "value": {
                           "start": 4,
                           "end": 21,
                           "text": "Amazon Rainforest",
                           "labels": ["LOC"],
                       },
                       "id": "j7gxEIzFc8",
                       "from_name": "label",
                       "to_name": "text",
                       "type": "labels",
                       "origin": "manual",
                   },
                   {
                       "value": {
                           "start": 64,
                           "end": 69,
                           "text": "world",
                           "labels": ["LOC"],
                       },
                       "id": "jvphR-ZYzf",
                       "from_name": "label",
                       "to_name": "text",
                       "type": "labels",
                       "origin": "manual",
                   },
                   {
                       "value": {
                           "start": 37,
                           "end": 56,
                           "text": "tropical rainforest",
                           "labels": ["LOC"],
                       },
                       "id": "dycisoes_7",
                       "from_name": "label",
                       "to_name": "text",
                       "type": "labels",
                       "origin": "manual",
                   },
               ],
               "was_cancelled": False,
               "ground_truth": False,
               "created_at": "2024-01-12T05:24:46.800863Z",
               "updated_at": "2024-01-12T05:41:38.833275Z",
               "draft_created_at": None,
               "lead_time": 20.229,
               "import_id": None,
               "last_action": None,
               "task": 55,
               "project": 2,
               "updated_by": 1,
               "parent_prediction": None,
               "parent_annotation": None,
               "last_created_by": None,
           }
       ],
       "predictions": [],
   },
   {
       "id": 56,
       "data": {"text": "Mount Everest is the highest mountain on Earth."},
       "meta": {},
       "created_at": "2024-01-11T20:21:52.446675Z",
       "updated_at": "2024-01-11T20:21:52.446687Z",
       "is_labeled": False,
       "overlap": 1,
       "inner_id": 6,
       "total_annotations": 0,
       "cancelled_annotations": 0,
       "total_predictions": 0,
       "comment_count": 0,
       "unresolved_comment_count": 0,
       "last_comment_updated_at": None,
       "project": 2,
       "updated_by": None,
       "file_upload": 13,
       "comment_authors": [],
       "annotations": [],
       "predictions": [],
   },
   {
       "id": 57,
       "data": {"text": "The Mona Lisa is a famous painting by Leonardo da Vinci."},
       "meta": {},
       "created_at": "2024-01-11T20:21:52.446744Z",
       "updated_at": "2024-01-11T20:21:52.446756Z",
       "is_labeled": False,
       "overlap": 1,
       "inner_id": 7,
       "total_annotations": 0,
       "cancelled_annotations": 0,
       "total_predictions": 0,
       "comment_count": 0,
       "unresolved_comment_count": 0,
       "last_comment_updated_at": None,
       "project": 2,
       "updated_by": None,
       "file_upload": 13,
       "comment_authors": [],
       "annotations": [],
       "predictions": [],
   },
   {
       "id": 58,
       "data": {"text": "Python is a popular programming language."},
       "meta": {},
       "created_at": "2024-01-11T20:21:52.446813Z",
       "updated_at": "2024-01-11T20:21:52.446825Z",
       "is_labeled": False,
       "overlap": 1,
       "inner_id": 8,
       "total_annotations": 0,
       "cancelled_annotations": 0,
       "total_predictions": 0,
       "comment_count": 0,
       "unresolved_comment_count": 0,
       "last_comment_updated_at": None,
       "project": 2,
       "updated_by": None,
       "file_upload": 13,
       "comment_authors": [],
       "annotations": [],
       "predictions": [],
   },
   {
       "id": 59,
       "data": {"text": "The Sahara Desert is the largest hot desert in the world."},
       "meta": {},
       "created_at": "2024-01-11T20:21:52.446881Z",
       "updated_at": "2024-01-11T20:21:52.446893Z",
       "is_labeled": False,
       "overlap": 1,
       "inner_id": 9,
       "total_annotations": 0,
       "cancelled_annotations": 0,
       "total_predictions": 0,
       "comment_count": 0,
       "unresolved_comment_count": 0,
       "last_comment_updated_at": None,
       "project": 2,
       "updated_by": None,
       "file_upload": 13,
       "comment_authors": [],
       "annotations": [],
       "predictions": [],
   },
   {
       "id": 60,
       "data": {
           "text": "The Statue of Liberty was a gift from France to the United States."
       },
       "meta": {},
       "created_at": "2024-01-11T20:21:52.446950Z",
       "updated_at": "2024-01-11T20:21:52.446962Z",
       "is_labeled": False,
       "overlap": 1,
       "inner_id": 10,
       "total_annotations": 0,
       "cancelled_annotations": 0,
       "total_predictions": 0,
       "comment_count": 0,
       "unresolved_comment_count": 0,
       "last_comment_updated_at": None,
       "project": 2,
       "updated_by": None,
       "file_upload": 13,
       "comment_authors": [],
       "annotations": [],
       "predictions": [],
   },
]
data = ""
model = NERModel()
model.fit("event", 0, data)

results = model.predict(tasks)
for task in range(len(tasks)):
   print(f"Task {tasks[task]['data']}\n")
   pretty_print_result(results[task])
   print("#" * 30)


# print(tasks[0]["data"]["text"])
# print(results[0]["result"])





