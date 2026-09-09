#!/usr/bin/env python3
"""Self-test of the records bucket and the signing key (run on the laptop, never on a box):

    ~/.config/nqueens/venv/bin/python tools/bucket_selftest.py [--bucket paw-nqueens-records] [--key ~/.config/nqueens/gcs-signer.json]

Checks the bucket's settings (uniform access, public access prevention, versioning), signs a ten-minute PUT URL
for runs/test/hello.csv, uploads two versions with curl exactly as a worker does, confirms that the PUT URL
cannot read, that the content reads back through the key, that both versions were kept, and removes the test
object again.  Prints nothing from the key beyond the service-account e-mail.
"""
import argparse, datetime, os, subprocess, tempfile
from google.cloud import storage
from google.oauth2 import service_account

ap = argparse.ArgumentParser()
ap.add_argument("--bucket", default="paw-nqueens-records")
ap.add_argument("--key", default=os.path.expanduser("~/.config/nqueens/gcs-signer.json"))
a = ap.parse_args()
creds = service_account.Credentials.from_service_account_file(a.key)
print("signer:", creds.service_account_email)
client = storage.Client(credentials=creds, project=creds.project_id)
b = client.get_bucket(a.bucket)
cfg = b.iam_configuration
print(f"bucket {b.name}: location {b.location}, versioning {b.versioning_enabled}, uniform access {cfg.uniform_bucket_level_access_enabled}, public access prevention {cfg.public_access_prevention}")
ok = True
if not b.versioning_enabled: print("  !! versioning is off — turn it on so an overwrite is recoverable"); ok = False
if not cfg.uniform_bucket_level_access_enabled: print("  !! uniform bucket-level access is off"); ok = False
if cfg.public_access_prevention != "enforced": print("  !! public access prevention is not enforced"); ok = False

blob = b.blob("runs/test/hello.csv")
url = blob.generate_signed_url(version="v4", expiration=datetime.timedelta(minutes=10), method="PUT")
with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f: path = f.name
for content in ("n,stride,phase\n8,2,0\n", "n,stride,phase\n8,2,0\n8,2,1\n"):
    open(path, "w").write(content)
    r = subprocess.run(["curl", "-sS", "-f", "-o", "/dev/null", "-w", "%{http_code}", "-X", "PUT", "-T", path, url], capture_output=True, text=True)
    print("PUT through the signed URL:", r.stdout or r.stderr.strip())
    if r.stdout != "200": ok = False
r = subprocess.run(["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", url], capture_output=True, text=True)
print("GET with the same URL (must be refused):", r.stdout)
if r.stdout == "200": ok = False
blob.reload()
versions = list(b.list_blobs(prefix="runs/test/hello.csv", versions=True))
print(f"object {blob.name}: {blob.size} bytes, {len(versions)} version(s) kept, content matches: {blob.download_as_text() == open(path).read()}")
if len(versions) < 2 or blob.download_as_text() != open(path).read(): ok = False
for v in versions: v.delete()
os.unlink(path)
print("test object removed;", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
