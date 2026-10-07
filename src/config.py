"""Read experiment settings from the frozen protocols, not a second config."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class StudyConfig:
    name: str
    seeds: tuple[int, ...]
    groups: tuple[int, ...]
    epochs: dict[int, int]
    architecture: tuple[int, ...]
    batch_size: int
    time_steps: int
    dt: float
    directory: Path

    def validate_group(self, group: int) -> None:
        # G=1 is the reference, so rejecting it would invalidate the comparison.
        if group not in self.groups:
            raise ValueError(f"G={group} is outside the frozen {self.name} study")
        if any(width % group for width in self.architecture[1:-1]):
            raise ValueError("Hidden-layer widths must divide into equal groups")


def read_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_config(name: str) -> StudyConfig:
    if name not in ("primary", "supplement"):
        raise ValueError("Choose primary or supplement")
    directory = ROOT / "studies" / name
    protocol = read_json(directory / "protocol.json")
    groups = tuple(protocol["groups"])
    epochs = ({group: protocol["epochs"] for group in groups}
              if name == "primary" else
              {int(group): count for group, count in protocol["epochs_by_group"].items()})
    config = StudyConfig(
        name, tuple(protocol["seeds"]), groups, epochs,
        tuple(protocol["architecture"]), protocol["batch"],
        protocol["T"], protocol["dt"], directory,
    )
    if not config.seeds or len(config.seeds) != len(set(config.seeds)):
        raise ValueError("The seed list must be nonempty and unique")
    for group in groups:
        config.validate_group(group)
    return config


def verify_sources(config: StudyConfig) -> None:
    directory = config.directory
    protocol = read_json(directory / "protocol.json")
    decision = read_json(directory / "freeze_decision.json")
    if sha256(directory / "protocol.json") != decision["protocol_sha256"]:
        raise ValueError("Frozen protocol hash changed")
    if sha256(directory / "source_manifest.json") != protocol["source_manifest_sha256"]:
        raise ValueError("Frozen source manifest changed")
    for filename, digest in read_json(directory / "source_manifest.json").items():
        if sha256(directory / "source" / filename) != digest:
            raise ValueError(f"Frozen model source changed: {filename}")
    runner = ("run_grouped_csdp_replication.py" if config.name == "primary"
              else "run_grouped_supplement.py")
    if sha256(ROOT / "src" / runner) != protocol["code_sha256"]:
        raise ValueError("Frozen training runner changed")
    if config.name == "supplement":
        if sha256(directory / "analysis_plan.json") != protocol["analysis_plan_sha256"]:
            raise ValueError("Frozen analysis plan changed")


def describe_configs() -> None:
    for name in ("primary", "supplement"):
        config = load_config(name)
        verify_sources(config)
        print(f"{name}: seeds={list(config.seeds)}, groups={list(config.groups)}, "
              f"epochs={config.epochs}")
    supplement = read_json(ROOT / "studies/supplement/protocol.json")
    print(f"Architecture: {supplement['architecture']}; raw batch: {supplement['batch']}; "
          f"contrastive batch: {supplement['effective_contrastive_batch']}; "
          f"T={supplement['T']}; dt={supplement['dt']}")
    print(f"Requested eta: {supplement['constructor_eta_requested']}; "
          f"actual eta: {supplement['actual_eta_w']}; "
          f"activity decay: {supplement['actual_weight_decay']}; "
          f"threshold: {supplement['theta']}")
