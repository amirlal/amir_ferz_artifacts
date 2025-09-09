# selden_vault.py
import json
import time
import hashlib

class SeldenVault:
    def __init__(self, protocol_name, steward="Amir", cycle_id=None):
        self.protocol_name = protocol_name
        self.steward = steward
        self.cycle_id = cycle_id or self._generate_cycle_id()
        self.records = []

    def _generate_cycle_id(self):
        ts = str(time.time())
        return hashlib.sha256(ts.encode()).hexdigest()[:12]

    def log_signal(self, signal_type, fidelity_score, decay_rate, notes=""):
        self.records.append({
            "cycle_id": self.cycle_id,
            "protocol": self.protocol_name,
            "signal_type": signal_type,
            "fidelity": round(fidelity_score, 4),
            "decay": round(decay_rate, 4),
            "notes": notes,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

    def export(self, path="selden_vault_log.json"):
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)
