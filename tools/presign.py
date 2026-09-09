#!/usr/bin/env python3
"""Signed upload URLs for the workers' records (RUNBOOK.md, "Records push").

Each box gets URLs that allow one thing: PUT to its own objects in the records bucket for at most
seven days.  The service-account key that signs them stays on this machine; only the URLs travel.

    presign.py N --stride S --hosts boxes.txt [--bucket paw-nqueens-records] [--key ~/.config/nqueens/gcs-signer.json]
               [--days 7] [--out push]

boxes.txt lines are "PORT HOST [LABEL]"; a missing LABEL becomes box01, box02, … in file order.  For each box
the file push/<LABEL>.env is written with PUSH_CSV_URL, PUSH_LOG_URL and PUSH_MANIFEST_URL; ship it to the box as
n-queens/runs/push_urls.env (tools/fleet.py does this).  Objects: runs/<N>_s<S>/parity_<N>_s<S>_<LABEL>.csv,
runs/<N>_s<S>/worker_<LABEL>.log, runs/<N>_s<S>/parity_<N>_s<S>_manifest.json.  Needs the google-cloud-storage
package (~/.config/nqueens/venv).  Signed URLs are valid for at most 7 days: renew and re-ship for longer runs.
"""
import argparse, datetime, os, sys
from google.cloud import storage
from google.oauth2 import service_account

ap = argparse.ArgumentParser()
ap.add_argument("n", type=int)
ap.add_argument("--stride", type=int, required=True)
ap.add_argument("--hosts", required=True)
ap.add_argument("--bucket", default="paw-nqueens-records")
ap.add_argument("--key", default=os.path.expanduser("~/.config/nqueens/gcs-signer.json"))
ap.add_argument("--days", type=float, default=7.0)
ap.add_argument("--out", default="push")
a = ap.parse_args()
if a.days > 7: sys.exit("V4 signed URLs are valid for at most 7 days")
creds = service_account.Credentials.from_service_account_file(a.key)
client = storage.Client(credentials=creds, project=creds.project_id)
bucket = client.bucket(a.bucket)
prefix = f"runs/{a.n}_s{a.stride}"
os.makedirs(a.out, exist_ok=True)
labels = []
with open(a.hosts) as f:
    for i, line in enumerate(l.strip() for l in f):
        if not line: continue
        parts = line.split()
        labels.append(parts[2] if len(parts) > 2 else f"box{i + 1:02d}")
exp = datetime.timedelta(days=a.days)
for label in labels:
    urls = {
        "PUSH_CSV_URL": bucket.blob(f"{prefix}/parity_{a.n}_s{a.stride}_{label}.csv").generate_signed_url(version="v4", expiration=exp, method="PUT"),
        "PUSH_LOG_URL": bucket.blob(f"{prefix}/worker_{label}.log").generate_signed_url(version="v4", expiration=exp, method="PUT"),
        "PUSH_MANIFEST_URL": bucket.blob(f"{prefix}/parity_{a.n}_s{a.stride}_manifest.json").generate_signed_url(version="v4", expiration=exp, method="PUT"),
    }
    path = os.path.join(a.out, f"{label}.env")
    with open(path, "w") as f:
        for k, v in urls.items(): f.write(f"{k}={v}\n")
    os.chmod(path, 0o600)
    print(f"{label}: {path} (PUT-only, {a.days:g} days, {prefix}/…_{label}.csv)")
print(f"{len(labels)} boxes; expiry {(datetime.datetime.now(datetime.timezone.utc) + exp).strftime('%Y-%m-%d %H:%M UTC')}")
