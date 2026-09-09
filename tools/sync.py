#!/usr/bin/env python3
"""Pull the records of a run from the bucket into a local directory (the replacement for ssh collection).

    sync.py N --stride S [--dir runs] [--bucket paw-nqueens-records] [--key ~/.config/nqueens/gcs-signer.json]

Downloads every object under runs/<N>_s<S>/ whose generation changed since the last sync (a small state file
in DIR remembers generations), flattening names into DIR so tools/merge.py finds them.  Needs the
google-cloud-storage package (~/.config/nqueens/venv); the key stays on this machine.
"""
import argparse, json, os
from google.cloud import storage
from google.oauth2 import service_account

ap = argparse.ArgumentParser()
ap.add_argument("n", type=int)
ap.add_argument("--stride", type=int, required=True)
ap.add_argument("--dir", default="runs")
ap.add_argument("--bucket", default="paw-nqueens-records")
ap.add_argument("--key", default=os.path.expanduser("~/.config/nqueens/gcs-signer.json"))
a = ap.parse_args()
creds = service_account.Credentials.from_service_account_file(a.key)
client = storage.Client(credentials=creds, project=creds.project_id)
prefix = f"runs/{a.n}_s{a.stride}/"
os.makedirs(a.dir, exist_ok=True)
state_path = os.path.join(a.dir, f".sync_{a.n}_s{a.stride}.json")
state = json.load(open(state_path)) if os.path.exists(state_path) else {}
seen, changed = 0, 0
for blob in client.list_blobs(a.bucket, prefix=prefix):
    name = blob.name[len(prefix):]
    if not name or "/" in name: continue
    seen += 1
    if state.get(name) == blob.generation: continue
    tmp = os.path.join(a.dir, name + ".part")
    blob.download_to_filename(tmp)
    os.replace(tmp, os.path.join(a.dir, name))
    state[name] = blob.generation
    changed += 1
    print(f"updated {name} ({blob.size} bytes, {blob.updated:%Y-%m-%d %H:%M:%S} UTC)")
json.dump(state, open(state_path, "w"))
print(f"{seen} objects under gs://{a.bucket}/{prefix}, {changed} updated")
