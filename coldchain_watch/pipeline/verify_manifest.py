"""
verify_manifest.py
------------------
Re-hashes every file recorded in manifest.csv and confirms
it matches the stored SHA-256.

Called by: make verify
Purpose: proves to the jury that no Bronze file was manually edited
after ingestion. Any mismatch means the file was tampered with.
"""

import sys
from pipeline.manifest import verify_all

if __name__ == "__main__":
    print("\nVerifying manifest.csv integrity...")
    ok = verify_all()
    if ok:
        print("\n✅ All files verified — manifest is clean.")
        sys.exit(0)
    else:
        print("\n❌ Verification failed — see above for details.")
        sys.exit(1)
