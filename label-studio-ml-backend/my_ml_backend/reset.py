from utils import LABEL_STUDIO_URL, LABEL_STUDIO_API_KEY
from label_studio_sdk import Client
import os
import init_model
import time

projectID_DEMO = 1
projectID_SOLO= 2
projectID_SOLO_CON= 3
projectID_PAIR= 4
projectID_PAIR_CON= 5

def inital_model_predictions():
    ls = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
    ls.check_connection()
    
    project = ls.get_project(projectID_DEMO)
    project.delete_all_tasks()
    project.import_tasks("project_tasks\demo.json")
    time.sleep(20)
    project.export_tasks(export_type='JSON', download_all_tasks=True, download_resources=True, export_location="project_tasks\demo.json")

    project = ls.get_project(projectID_SOLO)
    project.delete_all_tasks()
    project.import_tasks("project_tasks\solo.json")
    time.sleep(20)
    project.export_tasks(export_type='JSON', download_all_tasks=True, download_resources=True, export_location="project_tasks\solo.json")

    project = ls.get_project(projectID_SOLO_CON)
    project.delete_all_tasks()
    project.import_tasks("project_tasks\solo_con.json")
    time.sleep(20)
    project.export_tasks(export_type='JSON', download_all_tasks=True, download_resources=True, export_location="project_tasks\solo_con.json")

    project = ls.get_project(projectID_PAIR)
    project.delete_all_tasks()
    project.import_tasks("project_tasks\pair.json")
    time.sleep(20)
    project.export_tasks(export_type='JSON', download_all_tasks=True, download_resources=True, export_location="project_tasks\pair.json")

    project = ls.get_project(projectID_PAIR_CON)
    project.delete_all_tasks()
    project.import_tasks("project_tasks\pair_con.json")
    time.sleep(20)
    project.export_tasks(export_type='JSON', download_all_tasks=True, download_resources=True, export_location="project_tasks\pair_con.json")

    init_model.reset_model()
def main(working_project, trial_number):
    ls = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
    ls.check_connection()

    if ( (working_project == projectID_SOLO) ):
        project = ls.get_project(projectID_SOLO)
        project.export_tasks(export_type='JSON', download_all_tasks=True, download_resources=True, export_location=f"annotation_results\{trial_number}_solo.json")
        project.delete_all_tasks()
        project.import_tasks("project_tasks\solo.json")

    elif ( (working_project == projectID_SOLO_CON) ):
        project = ls.get_project(projectID_SOLO_CON)
        project.export_tasks(export_type='JSON', download_all_tasks=True, download_resources=True, export_location=f"annotation_results\{trial_number}_solo_con.json")
        project.delete_all_tasks()
        project.import_tasks("project_tasks\solo_con.json")

    elif ( (working_project == projectID_PAIR) ):
        project = ls.get_project(projectID_PAIR)
        project.export_tasks(export_type='JSON', download_all_tasks=True, download_resources=True, export_location=f"annotation_results\{trial_number}_pair.json")
        project.delete_all_tasks()
        project.import_tasks("project_tasks\pair.json")

    elif ( (working_project == projectID_PAIR_CON) ):
        project = ls.get_project(projectID_PAIR_CON)
        project.export_tasks(export_type='JSON', download_all_tasks=True, download_resources=True, export_location=f"annotation_results\{trial_number}_pair_con.json")
        project.delete_all_tasks()
        project.import_tasks("project_tasks\pair_con.json")

    else:
        print("Invalid project ID or delete_all_tasks flag")
        return

if __name__ == "__main__":
    '''
    reset_all_projects = input("Do you want to reset all projects? (y/n): ")
    trial_number = input("Enter the result's trial number: ")
    if reset_all_projects.lower().strip() == "y":
        all_projects = [projectID_SOLO, projectID_SOLO_CON, projectID_PAIR, projectID_PAIR_CON]
        for project in all_projects:
            main(project,  trial_number)
        init_model.reset_model()
    else:
        reset_project = input("Which project do you want to reset? (1: Demo, 2: Solo, 3: Solo_con, 4: Pair, 5: Pair_con): ")
        main(int(reset_project), trial_number)
        '''
    
    ls = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
    ls.check_connection()
    trial_number = projectID_SOLO
    project = ls.get_project(projectID_SOLO)
    project.export_tasks(export_type='JSON', download_all_tasks=True, download_resources=True, export_location=f"annotation_results\{trial_number}_solo.json")



