import csv
import time
import os
import zmq
import signal
from pylsl import StreamInlet, resolve_streams, local_clock
import xml.etree.ElementTree as ET

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
                        filtered_streams[f"{stream_name}_{participant_id}"] = (
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
        raise Exception(f"Failed to start recording: {message}")


def stop_pupil_capture_recording(remote_address):
    success, message = zmq_communicate(remote_address, "r", expect_reply=True)
    if success:
        print(f"Recording stopped: {message}")
    else:
        print(f"Failed to stop recording: {message}")
        raise Exception(f"Failed to stop recording: {message}") 
