import pylsl
from pylsl import (
    StreamInlet,
    resolve_stream,
    resolve_streams,
    TimeoutError,
    local_clock,
)
import xml.etree.ElementTree as ET

relevent_streams = [
    "pupil_capture",
    "pupil_capture_fixations",
    "pupil_capture_pupillometry_only",
    "ECL",
    "GSR2",
    "GSR1",
]


def try_pull_sample(inlet, attempts=1, timeout=5):
    for attempt in range(attempts):
        try:
            sample, timestamp = inlet.pull_sample(timeout=timeout)
            if sample:
                print(f"Successfully received sample from {inlet.info().name()}")
                return sample, timestamp
        except TimeoutError:
            print(
                f"Attempt {attempt+1}: No sample received for {inlet.info().name()} within timeout."
            )
    return None, None


def print_available_streams(verbose=False):
    print("Detecting all available LSL streams...")

    if not verbose:
        streams = resolve_streams()
        if streams:
            print(f"Found {len(streams)} streams:")
            for stream in streams:
                print(f"- {stream.name()} [{stream.type()}] from {stream.hostname()}")
        else:
            print("No streams found.")
        return

    else:
        try:
            streams = resolve_stream()  # Attempt to resolve all streams
            print(f"Found {len(streams)} stream(s).")
        except Exception as e:
            print(f"Error resolving streams: {e}")
            return

        if len(streams) == 0:
            print("No LSL streams detected.")
            return

        for i, stream in enumerate(streams):
            try:
                if stream.name() not in relevent_streams:
                    print(f"Skipping irrelevant stream: {stream.name()}")
                    print("#" * 50)
                    continue
                inlet = StreamInlet(stream)
                info = inlet.info()
                xml_root = ET.fromstring(info.as_xml())
                print(f"Stream {i+1}:")
                print(f"  Name: {info.name()}")
                print(f"  Type: {info.type()}")
                hostname_element = xml_root.find("./hostname")
                hostname = (
                    hostname_element.text if hostname_element is not None else "N/A"
                )
                print(f"  Hostname: {hostname}")
                print(f"  Source ID: {info.source_id()}")
                sample, timestamp = try_pull_sample(inlet)
                if sample:
                    print(f"  Sample Packet: {sample}")
                else:
                    print("  No sample received within timeout.")
            except Exception as e:
                print(f"  Error with stream {i+1}: {e}")
            print("#" * 50)


if __name__ == "__main__":
    print_available_streams(True)
