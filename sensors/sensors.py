import csv
import time
import os
import zmq
import signal
from pylsl import StreamInlet, resolve_streams, local_clock
import xml.etree.ElementTree as ET
import sensor_util

# Signal handler for graceful shutdown
stop_loop = False


def signal_handler(sig, frame):
    global stop_loop
    stop_loop = True


signal.signal(signal.SIGINT, signal_handler)

# Remote addresses
huggin_remote_address = "129.21.230.120"
midgard_remote_address = "129.21.230.122"

# Data labels for pupil capture and GSR
pupil_capture_labels = [
    "confidence",
    "norm_pos_x",
    "norm_pos_y",
    "gaze_point_3d_x",
    "gaze_point_3d_y",
    "gaze_point_3d_z",
    "eye_center0_3d_x",
    "eye_center0_3d_y",
    "eye_center0_3d_z",
    "eye_center1_3d_x",
    "eye_center1_3d_y",
    "eye_center1_3d_z",
    "gaze_normal0_x",
    "gaze_normal0_y",
    "gaze_normal0_z",
    "gaze_normal1_x",
    "gaze_normal1_y",
    "gaze_normal1_z",
    "diameter0_2d",
    "diameter1_2d",
    "diameter0_3d",
    "diameter1_3d",
]

GSR_labels = [
    "Timestamp RAW (no units)",
    "Timestamp CAL (mSecs)",
    "System Timestamp CAL (mSecs)",
    "Wide Range Accelerometer X RAW (no units)",
    "Wide Range Accelerometer X CAL (m/(sec ^2))",
    "Wide Range Accelerometer Y RAW (no units)",
    "Wide Range Accelerometer Y CAL (m/(sec ^2))",
    "Wide Range Accelerometer Z RAW (no units)",
    "Wide Range Accelerometer Z CAL (m/(sec ^2))",
    "VSenseBatt RAW (no units)",
    "VSenseBatt CAL (m Volts)",
    "Internal ADC A13 RAW (no units)",
    "Internal ADC A13 CAL (m Volts)",
    "GSR RAW (no units)",
    "GSR CAL (kOhms)",
    "GSR Conductance CAL (u Siemens)",
]


def main(participants, trial_name, record=False):
    hostnames = {"midgard": participants[0], "huginn": participants[1]}
    stream_names = {
        "pupil_capture": pupil_capture_labels,
        "GSR1": GSR_labels,
        "GSR2": GSR_labels,
    }
    inlets = {}

    # Ensure directories exist
    recording_dir = os.path.join("recordings", time.strftime("%Y%m%d-%H%M%S"))
    collected_data_dir = "collected_data"
    os.makedirs(recording_dir, exist_ok=True)
    os.makedirs(collected_data_dir, exist_ok=True)

    # Collect and filter streams
    inlets = sensor_util.collect_and_filter_streams(stream_names, hostnames)

    print("Resolved streams\n\n\n")
    if record:
        if hostnames["midgard"] != "---":
            print("Starting recording on Midgard...")
            sensor_util.start_pupil_capture_recording(
                recording_dir, midgard_remote_address
            )
        if hostnames["huginn"] != "---":
            print("Starting recording on Huggin...")
            sensor_util.start_pupil_capture_recording(
                recording_dir, huggin_remote_address
            )

    input("Press Enter to start data collection...")

    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{trial_name}_{participants[0]}_{participants[1]}_{timestamp}.csv"
        with open(
            os.path.join(collected_data_dir, filename), "w", newline=""
        ) as csvfile:
            fieldnames = ["timestamp"]
            # add the stream names to the fieldnames
            for stream_name, (inlet, labels, participant_id) in inlets.items():
                for label in labels:
                    print(f"{label}_{participant_id}")
                    fieldnames.append(f"{label}_{participant_id}")

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            print("Starting data collection...")
            target_duration = 1 / 15

            while not stop_loop:  # Use the stop_loop flag to exit the loop gracefully
                iter_start = time.time()
                row = {"timestamp": local_clock()}
                for stream_name, (inlet, labels, participant_id) in inlets.items():

                    sample, timestamp = inlet.pull_sample(timeout=1.0)
                    if sample:
                        for i, label in enumerate(labels):
                            row[f"{label}_{participant_id}"] = sample[i]
                    else:
                        print(
                            f"No data received from {stream_name} within timeout period."
                        )

                writer.writerow(row)
                iter_end = time.time()
                iter_duration = iter_end - iter_start
                sleep_duration = max(0, target_duration - iter_duration)
                time.sleep(sleep_duration)
    finally:
        csvfile.close()
        print("Data collection stopped.")
        if record:
            if hostnames["midgard"] != "---":
                sensor_util.stop_pupil_capture_recording(midgard_remote_address)
                print(
                    f"Recording stopped on Midgard. Recording saved to {recording_dir}"
                )
            if hostnames["huginn"] != "---":
                sensor_util.stop_pupil_capture_recording(huggin_remote_address)
                print(
                    f"Recording stopped on Huggin. Recording saved to {recording_dir}"
                )


if __name__ == "__main__":
    participants_input = input(
        "Enter the participant IDs to record separated by comma (e.g., Participant_1,Participant_2): "
    )
    participants = [pid.strip() for pid in participants_input.split(",")]
    trial_name = input("Enter the trial name: ")
    main(participants, trial_name, record=True)
