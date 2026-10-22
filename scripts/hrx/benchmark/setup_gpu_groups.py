#!/usr/bin/env python3
"""Give the runner account access to the GPU devices mounted by its host.

Container group databases can omit the host's device GIDs. Benchmark commands
must start a fresh user session after this setup to acquire the added groups.
"""

import grp
import os
from pathlib import Path
import pwd
import subprocess


def main() -> None:
    user = pwd.getpwuid(os.getuid()).pw_name
    devices = [Path("/dev/kfd"), *Path("/dev/dri").glob("renderD*")]
    # /dev/kfd is required for HRX; fail here if the runner lacks the mount.
    gids = sorted({device.stat().st_gid for device in devices})
    for gid in gids:
        try:
            group = grp.getgrgid(gid).gr_name
        except KeyError:
            group = f"ci-gpu-{gid}"
            subprocess.run(["sudo", "-n", "groupadd", "--gid", str(gid), group], check=True)
        subprocess.run(["sudo", "-n", "usermod", "--append", "--groups", group, user], check=True)
        print(f"Added {user} to GPU device group {group} ({gid})", flush=True)


if __name__ == "__main__":
    main()
