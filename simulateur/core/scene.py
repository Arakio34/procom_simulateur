from dataclasses import dataclass, field, asdict

from .exceptions import ValidationError


@dataclass
class Layer:
    z_min: float
    z_max: float
    c: float
    rho: float
    name: str = "layer"

    def validate(self, index):
        errors = []
        if self.z_min < 0:
            errors.append("z_min must be >= 0")
        if self.z_max <= self.z_min:
            errors.append("z_max must be > z_min")
        if self.c <= 0:
            errors.append("c must be > 0")
        if self.rho <= 0:
            errors.append("rho must be > 0")
        if errors:
            raise ValidationError(
                f"Invalid layer[{index}]: " + "; ".join(errors)
            )

    def to_dict(self):
        return asdict(self)


@dataclass
class Scene:
    points: list
    layers: list = field(default_factory=list)

    @classmethod
    def from_json(cls, data):
        if not isinstance(data, dict):
            raise ValidationError("Scene JSON must be an object.")
        unknown = sorted(set(data.keys()) - {"points", "layers"})
        if unknown:
            raise ValidationError(f"Unknown scene keys: {', '.join(unknown)}")
        points = data.get("points", [])
        layers = data.get("layers", [])
        if not isinstance(points, list):
            raise ValidationError("Scene points must be a list.")
        if not isinstance(layers, list):
            raise ValidationError("Scene layers must be a list.")
        cleaned_points = []
        for i, point in enumerate(points):
            if (
                not isinstance(point, list)
                or len(point) != 3
            ):
                raise ValidationError(
                    f"Point[{i}] must be a list of 3 values."
                )
            try:
                cleaned_points.append(
                    [float(point[0]), float(point[1]), float(point[2])]
                )
            except (TypeError, ValueError):
                raise ValidationError(
                    f"Point[{i}] values must be numeric."
                )
        layer_objs = []
        for i, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise ValidationError(f"Layer[{i}] must be an object.")
            missing = {"z_min", "z_max", "c", "rho"} - set(layer.keys())
            if missing:
                raise ValidationError(
                    f"Layer[{i}] missing keys: {', '.join(sorted(missing))}"
                )
            layer_objs.append(
                Layer(
                    z_min=float(layer["z_min"]),
                    z_max=float(layer["z_max"]),
                    c=float(layer["c"]),
                    rho=float(layer["rho"]),
                    name=layer.get("name", f"layer_{i}"),
                )
            )
        return cls(points=cleaned_points, layers=layer_objs)

    @classmethod
    def from_json_file(cls, path):
        import json
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_json(json.load(handle))

    def validate(self):
        if not isinstance(self.points, list):
            raise ValidationError("Scene points must be a list.")
        for i, point in enumerate(self.points):
            if (
                not isinstance(point, list)
                or len(point) != 3
            ):
                raise ValidationError(
                    f"Point[{i}] must be a list of 3 values."
                )
            try:
                float(point[0])
                float(point[1])
                float(point[2])
            except (TypeError, ValueError):
                raise ValidationError(
                    f"Point[{i}] values must be numeric."
                )
        for i, layer in enumerate(self.layers):
            if not isinstance(layer, Layer):
                raise ValidationError(f"Layer[{i}] must be a Layer.")
            layer.validate(i)

    @classmethod
    def random(cls, rng, max_point, x_range, z_range, amp_range=(1.0, 3.0)):
        if max_point <= 0:
            raise ValidationError("max_point must be > 0 for random scene.")
        nb_points = int(rng.integers(1, max_point + 1))
        points = []
        for _ in range(nb_points):
            points.append(
                [
                    float(rng.uniform(x_range[0], x_range[1])),
                    float(rng.uniform(z_range[0], z_range[1])),
                    float(rng.uniform(amp_range[0], amp_range[1])),
                ]
            )
        return cls(points=points, layers=[])

    def to_dict(self):
        return {
            "points": self.points,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    def summary(self):
        lines = []
        lines.append(f"points: {len(self.points)}")
        lines.append(f"layers: {len(self.layers)}")
        for layer in self.layers:
            lines.append(
                f"  - {layer.name}: {layer.z_min}..{layer.z_max} m, c={layer.c}, rho={layer.rho}"
            )
        return "\n".join(lines)
