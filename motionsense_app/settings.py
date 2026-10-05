import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root: Path
    state_root: Path | None = None

    @property
    def instance_marker(self) -> str:
        paths = [os.path.normcase(str(path.resolve()))
                 for path in (self.root, self.state_root or self.root)]
        return hashlib.sha256(json.dumps(paths, ensure_ascii=False).encode('utf-8')).hexdigest()

    @property
    def var_root(self) -> Path:
        return (self.state_root or self.root) / "var"

    @property
    def data_dir(self) -> Path:
        return self.var_root / "datasets"

    @property
    def model_dir(self) -> Path:
        return self.var_root / "models"

    @property
    def db_path(self) -> Path:
        return self.var_root / "motionsense.sqlite3"

    @property
    def frontend_dir(self) -> Path:
        return self.root / "frontend" / "dist"

    @classmethod
    def default(cls, root: Path | None = None) -> "Settings":
        root = (root or Path(__file__).resolve().parents[1]).resolve()
        state_root = os.environ.get("MOTIONSENSE_STATE_ROOT")
        state = Path(state_root) if state_root else None
        if state is not None:
            state = (state if state.is_absolute() else root / state).resolve()
        return cls(root=root, state_root=state)
