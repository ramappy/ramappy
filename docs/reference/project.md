# Project Model

`ramappy.project` groups multiple `SpectralMap` datasets together with shared metadata into a single project, and manages the serialization format and versioning for project files (primarily used by the RamApp frontend).

For the full auto-generated API reference see {py:mod}`ramappy.project`.

---

## Project

{py:mod}`ramappy.project.project` exposes the top-level project model that groups multiple `SpectralMap` datasets together with shared metadata such as project name, creation date, and author information.

**Full reference:** {py:mod}`ramappy.project.project`

---

## Versions

{py:mod}`ramappy.project.versions` handles the serialization format versioning for project files, providing migration logic for loading older project formats.

**Full reference:** {py:mod}`ramappy.project.versions`
