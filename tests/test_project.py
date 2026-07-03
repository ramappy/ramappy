import datetime

import numpy as np

from ramappy.core import SpectralMap
from ramappy.pipeline.pipeline import Pipeline
from ramappy.project.project import RamAppProject


def test_project_initialization():
    x = np.linspace(100, 1800, 10)
    intensities = np.zeros((4, 10))
    hsi = SpectralMap(x=x, data=intensities, img_width=2, img_height=2)
    pipeline = Pipeline()

    project = RamAppProject(data=hsi, pipeline=pipeline, project_id="test-proj")

    assert project.project_id == "test-proj"
    assert project.data is hsi
    assert project.pipeline is pipeline
    assert isinstance(project.last_access, datetime.datetime)
    assert hasattr(hsi, "_project")
    assert hsi._project is project


def test_project_export_pipeline(tmp_path):
    x = np.linspace(100, 1800, 10)
    intensities = np.zeros((4, 10))
    hsi = SpectralMap(x=x, data=intensities, img_width=2, img_height=2)
    pipeline = Pipeline(input_format="nexus")

    project = RamAppProject(data=hsi, pipeline=pipeline, project_id="test")

    yaml_path = tmp_path / "p.yaml"
    project.export_pipeline(yaml_path)
    assert yaml_path.exists()
    assert "input_format: nexus" in yaml_path.read_text()
