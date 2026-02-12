from dataclasses import dataclass, asdict

from .exceptions import ValidationError


@dataclass
class Parameters:
    c: float = 1540.0
    f0: float = 5e6
    fs: float = 40e6
    fracBW: float = 0.6
    nCycles: float = 2.5
    Nelem: int = 80
    pitch: float = 0.15e-3
    x_span: float = 20e-3
    z_min: float = 10e-3
    z_max: float = 50e-3
    Nx: int = 256
    Nz: int = 256
    SNR_dB: float = 15.0
    p: float = 985.0

    @classmethod
    def from_json(cls, data):
        if not isinstance(data, dict):
            raise ValidationError("Parameters JSON must be an object.")
        field_names = set(cls.__dataclass_fields__.keys())
        unknown = sorted(set(data.keys()) - field_names)
        if unknown:
            raise ValidationError(f"Unknown parameter keys: {', '.join(unknown)}")
        cleaned = {}
        for key in field_names:
            if key not in data:
                continue
            value = data[key]
            if key in {"Nelem", "Nx", "Nz"}:
                try:
                    cleaned[key] = int(value)
                except (TypeError, ValueError):
                    raise ValidationError(f"{key} must be an integer")
            else:
                try:
                    cleaned[key] = float(value)
                except (TypeError, ValueError):
                    raise ValidationError(f"{key} must be a number")
        return cls(**cleaned)

    @classmethod
    def from_json_file(cls, path):
        import json
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_json(json.load(handle))

    def validate(self):
        errors = []
        if not isinstance(self.Nelem, int):
            errors.append("Nelem must be an integer")
        if not isinstance(self.Nx, int) or not isinstance(self.Nz, int):
            errors.append("Nx and Nz must be integers")
        if self.c <= 0:
            errors.append("c must be > 0")
        if self.f0 <= 0:
            errors.append("f0 must be > 0")
        if self.fs <= 0:
            errors.append("fs must be > 0")
        if self.Nelem <= 0:
            errors.append("Nelem must be > 0")
        if self.pitch <= 0:
            errors.append("pitch must be > 0")
        if self.x_span <= 0:
            errors.append("x_span must be > 0")
        if self.z_min < 0:
            errors.append("z_min must be >= 0")
        if self.z_max <= self.z_min:
            errors.append("z_max must be > z_min")
        if self.Nx <= 0 or self.Nz <= 0:
            errors.append("Nx and Nz must be > 0")
        if self.p <= 0:
            errors.append("p must be > 0")
        if errors:
            raise ValidationError("Invalid parameters: " + "; ".join(errors))

    def with_overrides(self, **overrides):
        data = asdict(self)
        for key, value in overrides.items():
            if value is None:
                continue
            if key not in data:
                raise ValidationError(f"Unknown parameter override: {key}")
            data[key] = value
        return Parameters(**data)

    def to_dict(self):
        return asdict(self)
