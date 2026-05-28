import os
os.environ.setdefault('LD_PRELOAD', '/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
import omni.replicator.core as rep
methods = [x for x in dir(rep.create) if "light" in x.lower()]
with open("/tmp/light_methods.txt", "w") as f:
    f.write(str(methods) + "\n")
    # Also try rep.create.light
    try:
        l = rep.create.light()
        f.write(f"rep.create.light() works: {l}\n")
    except Exception as e:
        f.write(f"rep.create.light() error: {e}\n")
app.close()
