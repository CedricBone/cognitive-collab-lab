import csv
import time
import os
import zmq
import signal
from pylsl import StreamInlet, resolve_streams, local_clock
import xml.etree.ElementTree as ET

# Signal handler for graceful shutdown
stop_loop = False


def signal_handler(sig, frame):
    global stop_loop
    stop_loop = True


signal.signal(signal.SIGINT, signal_handler)

# Remote addresses
GOL3600_remote_address = "10.115.255.254"
midgard_remote_address = "129.21.230.214"

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


def collect_and_filter_streams(stream_names, hostnames):
    all_streams = resolve_streams()
    filtered_streams = {}
    for stream in all_streams:
        stream_hostname = stream.hostname()
        stream_name = stream.name()
        for hostname, participant_id in hostnames.items():
            if (stream_hostname == hostname) and (stream_name in stream_names.keys()):
                try:
                    inlet = StreamInlet(stream)
                    sample, timestamp = inlet.pull_sample(timeout=0.5)
                    if sample:
                        if stream_name == "GSR2":
                            filtered_streams[f"{stream_name}"] = (
                                inlet,
                                stream_names[stream_name],
                                "002",
                            )
                            print(
                            f"Stream {stream_name} from {stream_hostname} (Participant 002) collected.")
                        else:
                            filtered_streams[f"{stream_name}"] = (
                                inlet,
                                stream_names[stream_name],
                                participant_id,
                            )
                            print(
                            f"Stream {stream_name} from {stream_hostname} (Participant {participant_id}) collected.")
                except Exception as e:
                    print(
                        f"Error creating inlet for stream {stream_name} from {stream_hostname}: {e}"
                    )
    return filtered_streams


def zmq_communicate(remote_address, message, expect_reply=True):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    pupil_remote_address = f"tcp://{remote_address}:50020"
    try:
        socket.connect(pupil_remote_address)
        socket.send_string(message)
        if expect_reply:
            reply = socket.recv_string()
            return True, reply
    except zmq.ZMQError as e:
        print(f"ZMQ communication error: {e}")
        return False, str(e)
    finally:
        socket.close()
        context.term()


def start_pupil_capture_recording(recording_path, remote_address):
    success, message = zmq_communicate(remote_address, f"R {recording_path}")
    if success:
        print(f"Recording started: {message}")
    else:
        print(f"Failed to start recording: {message}")


def stop_pupil_capture_recording(remote_address):
    success, message = zmq_communicate(remote_address, "r", expect_reply=True)
    if success:
        print(f"Recording stopped: {message}")
    else:
        print(f"Failed to stop recording: {message}")


def main(participants, trial_name, record=False):
    hostnames = {"midgard": participants[0], "DESKTOP-972NVCE": participants[1]}
    stream_names = {
        "pupil_capture": pupil_capture_labels,
        "GSR1": GSR_labels,
        "GSR2": GSR_labels,
    }
    inlets = {}

    # Ensure directories exist
    recording_dir = os.path.join("recordings", trial_name)
    collected_data_dir = "collected_data"
    os.makedirs(recording_dir, exist_ok=True)
    os.makedirs(collected_data_dir, exist_ok=True)

    # Collect and filter streams
    inlets = collect_and_filter_streams(stream_names, hostnames)

    print("Resolved streams\n\n\n")
    if record:
        start_pupil_capture_recording(recording_dir, GOL3600_remote_address)
        start_pupil_capture_recording(recording_dir, midgard_remote_address)

    input("Press Enter to start data collection...")

    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{trial_name}_{timestamp}.csv"
        with open(
            os.path.join(collected_data_dir, filename), "w", newline=""
        ) as csvfile:
            fieldnames = ["timestamp"]
            print(f"inlets: {inlets}")
            # inlets = [f"{stream_name}_{participant_id}"] = (inlet,stream_names[stream_name],participant_id,)
            # add the stream names to the fieldnames
            for stream_name, (inlet, labels, participant_id) in inlets.items():
                for label in labels:
                    fieldnames.append(f"{label}_{participant_id}")

            print(f"fieldnames: {fieldnames}")
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            print("Starting data collection...")
            while not stop_loop:  # Use the stop_loop flag to exit the loop gracefully
                row = {"timestamp": local_clock()}
                for stream_name, (inlet, labels, participant_id) in inlets.items():
                    sample, timestamp = inlet.pull_sample(timeout=1.0)
                    if sample:
                        # Prefix each label with the participant ID to match the fieldnames
                        row.update(
                            {
                                f"{label}_{participant_id}": value
                                for label, value in zip(labels, sample)
                            }
                        )
                    else:
                        print(
                            f"No data received from {stream_name} within timeout period."
                        )

                writer.writerow(row)
                time.sleep(0.1)  # Adjust based on the required sampling rate
    finally:
        csvfile.close()
        print("Data collection stopped.")
        if record:
            stop_pupil_capture_recording(GOL3600_remote_address)
            stop_pupil_capture_recording(midgard_remote_address)


if __name__ == "__main__":
    participants_input = input(
        "Enter the participant IDs to record separated by comma (e.g., Participant_1,Participant_2): "
    )
    participants = [pid.strip() for pid in participants_input.split(",")]
    trial_name = input("Enter the trial name: ")
    main(participants, trial_name, record=False)
