from dataclasses import dataclass, field


@dataclass
class AssetVersions:
    """Tracks version numbers for various entities to handle cache invalidation.
    Each map stores entity_id -> version_number.
    """

    images: dict[str, int] = field(default_factory=dict)
    masks: dict[str, int] = field(default_factory=dict)
    spectra: dict[str, int] = field(default_factory=dict)

    def bump(self, entity_type: str, entity_id: str) -> int:
        """Increment version for a specific entity. Returns the new version."""
        if entity_type not in ["images", "masks", "spectra"]:
            raise ValueError(f"Invalid entity type: {entity_type}")

        target_dict = getattr(self, entity_type)
        new_version = target_dict.get(entity_id, 0) + 1
        target_dict[entity_id] = new_version
        return new_version

    def get(self, entity_type: str, entity_id: str) -> int:
        """Get the current version of an entity (default 0)."""
        target_dict = getattr(self, entity_type)
        return target_dict.get(entity_id, 0)
