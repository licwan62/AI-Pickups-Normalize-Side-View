from pickup_measure.src.settings import load_settings


def test_loads_output_ppi(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("output:\n  ppi: 72\n", encoding="utf-8")

    assert load_settings(config).output.ppi == 72


def test_loads_annotation_style(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "annotation:\n  outline_width_mm: 10\n  font_size_mm: 90\n  image_opacity: 0.5\n",
        encoding="utf-8",
    )
    settings = load_settings(config)
    assert settings.annotation.outline_width_mm == 10
    assert settings.annotation.font_size_mm == 90


def test_loads_quality_thresholds(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "quality:\n"
        "  pass_max_percent: 3\n"
        "  warning_max_percent: 15\n"
        "  error_max_percent: 20\n",
        encoding="utf-8",
    )

    settings = load_settings(config)

    assert settings.quality.pass_max_percent == 3
    assert settings.quality.warning_max_percent == 15
    assert settings.quality.error_max_percent == 20


def test_quality_error_threshold_defaults_to_warning_threshold(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "quality:\n  pass_max_percent: 3\n  warning_max_percent: 15\n",
        encoding="utf-8",
    )

    settings = load_settings(config)

    assert settings.quality.error_max_percent == 15
