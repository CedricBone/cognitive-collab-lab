from utils import LABEL_STUDIO_URL, LABEL_STUDIO_API_KEY
from label_studio_sdk import Client
import os
import init_model
import time
import json

projectID_DEMO = 1
projectID_SOLO= 2
projectID_SOLO_CON= 3
projectID_PAIR= 4
projectID_PAIR_CON= 5

def reformat_exort_json(file_path):
    with open(f"annotation_results\{file_path}.json", "r") as file:
        original_data = json.load(file)
        
    transformed_data = [{"text": item["data"]["text"]} for item in original_data]

    transformed_json = json.dumps(transformed_data, indent=2)

    with open(f"annotation_results\reformatted_{file_path}.json", "w") as file:
        file.write(transformed_json)

def collect_project_data(projectID, trial_type, participant_id):
    ls = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
    ls.check_connection()
    
    project = ls.get_project(projectID)
    data = project.get_paginated_tasks(filters=None, ordering=None, view_id=None, selected_ids=None, page=1, page_size=-1, only_ids=False, resolve_uri=True)
    if not os.path.exists(f"annotation_results\{trial_type}"):
        os.makedirs(f"annotation_results\{trial_type}")
    with open(f"annotation_results\{trial_type}\{trial_type}_{participant_id}_{time.strftime('%Y%m%d-%H%M%S')}.json", "w") as file:
        json.dump(data, file, indent=2)


def main(working_project, trial_type, participant_id):
    ls = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
    ls.check_connection()

    if ( (working_project == projectID_SOLO) ):
        project = ls.get_project(projectID_SOLO)
        collect_project_data(projectID_SOLO, trial_type, participant_id)
        project.delete_all_tasks()
        project.import_tasks("project_tasks\solo.json")
        print("Solo project reset")
    elif ( (working_project == projectID_SOLO_CON) ):
        project = ls.get_project(projectID_SOLO_CON)
        collect_project_data(projectID_SOLO_CON, trial_type, participant_id)
        project.delete_all_tasks()
        project.import_tasks("project_tasks\solo_con.json")
        print("Solo_con project reset")
    elif ( (working_project == projectID_PAIR) ):
        project = ls.get_project(projectID_PAIR)
        collect_project_data(projectID_PAIR, trial_type, participant_id)
        project.delete_all_tasks()
        project.import_tasks("project_tasks\pair.json")
        print("Pair project reset")
    elif ( (working_project == projectID_PAIR_CON) ):
        project = ls.get_project(projectID_PAIR_CON)
        collect_project_data(projectID_PAIR_CON, trial_type, participant_id)
        project.delete_all_tasks()
        project.import_tasks("project_tasks\pair_con.json")
        print("Pair_con project reset")
    else:
        print("Invalid project ID")
        return

if __name__ == "__main__":
    projects = {1: "Demo", 2: "Solo", 3: "Solo_con", 4: "Pair", 5: "Pair_con"}
    reset_all_projects = input("Do you want to reset all projects? (y/n): ")
    participant_id = input("Enter the participant ID: ")

    if reset_all_projects.lower().strip() == "y":
        all_projects = [projectID_SOLO, projectID_SOLO_CON, projectID_PAIR, projectID_PAIR_CON]
        for project in all_projects:
            main(project,  projects[project], participant_id)
        init_model.move_model(projects[project], participant_id)
        init_model.reset_model()
    else:
        reset_project = int(input("Which project do you want to reset? (1: Demo, 2: Solo, 3: Solo_con, 4: Pair, 5: Pair_con): "))
        main(int(reset_project), projects[reset_project], participant_id)
        init_model.move_model(projects[reset_project], participant_id)
        init_model.reset_model()








