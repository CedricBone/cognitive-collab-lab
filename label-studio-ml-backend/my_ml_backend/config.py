# Configuration variables
LABEL_STUDIO_URL = "http://localhost:8080"
#LABEL_STUDIO_API_KEY = "5e185926c1b42768a90baf72ae9d13994dab25b0"
LABEL_STUDIO_API_KEY = "ac5503be774a6e6ad2c4d8c031bdef2543300cc4"
base_url = "http://localhost:8080/api/predictions/"
#headers = {"Authorization": "Token 5e185926c1b42768a90baf72ae9d13994dab25b0"}
headers = {"Authorization": "Token ac5503be774a6e6ad2c4d8c031bdef2543300cc4"}
projectID = 1
#LABELS = ["PER", "ORG", "LOC", "MISC"]
LABELS = ['Person', 'Location', 'Organization', 'Miscellaneous']
model_version = "CLaSP_v1"
