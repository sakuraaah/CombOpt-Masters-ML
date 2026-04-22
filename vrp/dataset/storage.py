import json
from dataclasses import asdict, dataclass
from pathlib import Path

from vrp.constants import SHARD_SIZE
from vrp.models.sample import Sample


@dataclass
class SampleIndexEntry:
    path: str
    offset: int
    sample_id: int


class TrainingDatasetStorage:
    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self.shards_dir = dataset_dir / "shards"
        self.progress_path = dataset_dir / "progress.json"
        self.manifest_path = dataset_dir / "manifest.json"
        self.shard_size = self.read_manifest_shard_size()
        if self.shard_size is None:
            self.shard_size = SHARD_SIZE

    def initialize(self, manifest: dict[str, object]) -> None:
        self.shards_dir.mkdir(parents=True, exist_ok=True)

        if not self.manifest_path.exists():
            self.manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )

        manifest_shard_size = manifest.get("shard_size")
        if manifest_shard_size is not None:
            self.shard_size = int(manifest_shard_size)

    def get_completed_sample_ids(self) -> set[int]:
        sample_ids: set[int] = set()

        if not self.shards_dir.exists():
            return sample_ids

        for shard_path in sorted(self.shards_dir.glob("samples-*.jsonl")):
            with shard_path.open("r", encoding="utf-8") as shard_file:
                for line in shard_file:
                    if line.strip():
                        record = json.loads(line)
                        sample_ids.add(int(record["sample_id"]))

        return sample_ids

    def append_sample(self, sample: Sample) -> None:
        sample_id = sample.sample_id
        shard_path = self.get_shard_path(sample_id)

        with shard_path.open("a", encoding="utf-8") as shard_file:
            shard_file.write(json.dumps(sample.to_dict()) + "\n")

    def load_sample(self, sample_id: int) -> Sample:
        shard_path = self.get_shard_path(sample_id)

        if not shard_path.exists():
            raise FileNotFoundError(f"Sample shard not found for sample_id={sample_id}")

        with shard_path.open("r", encoding="utf-8") as shard_file:
            for line in shard_file:
                if not line.strip():
                    continue

                record = json.loads(line)
                if int(record["sample_id"]) == sample_id:
                    return Sample.from_dict(record)

        raise ValueError(f"Sample with sample_id={sample_id} was not found")

    def build_or_load_index(self) -> list[SampleIndexEntry]:
        index_path = self.dataset_dir / "index_offsets.json"

        if index_path.exists():
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            return [SampleIndexEntry(**item) for item in payload]

        entries: list[SampleIndexEntry] = []

        if not self.shards_dir.exists():
            return entries

        for shard_path in sorted(self.shards_dir.glob("samples-*.jsonl")):
            with shard_path.open("rb") as shard_file:
                while True:
                    offset = shard_file.tell()
                    line = shard_file.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue

                    record = json.loads(line)
                    entries.append(
                        SampleIndexEntry(
                            path=str(shard_path),
                            offset=offset,
                            sample_id=int(record["sample_id"]),
                        )
                    )

        index_path.write_text(
            json.dumps([asdict(entry) for entry in entries], indent=2),
            encoding="utf-8",
        )
        return entries

    def read_indexed_record(self, entry: SampleIndexEntry) -> dict[str, object]:
        with Path(entry.path).open("rb") as shard_file:
            shard_file.seek(entry.offset)
            line = shard_file.readline()

        return json.loads(line)

    def write_progress(self, target_sample_count: int, completed_sample_ids: set[int]) -> None:
        progress = {
            "target_sample_count": target_sample_count,
            "completed_sample_count": len(completed_sample_ids),
            "next_sample_id": self.get_next_sample_id(completed_sample_ids),
        }
        self.progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    def get_next_sample_id(self, completed_sample_ids: set[int]) -> int:
        next_sample_id = 0
        while next_sample_id in completed_sample_ids:
            next_sample_id += 1
        return next_sample_id

    def get_shard_path(self, sample_id: int) -> Path:
        shard_index = sample_id // self.shard_size
        return self.shards_dir / f"samples-{shard_index:05d}.jsonl"

    def read_manifest_shard_size(self) -> int | None:
        if not self.manifest_path.exists():
            return None

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        shard_size = manifest.get("shard_size")
        if shard_size is None:
            return None

        return int(shard_size)
