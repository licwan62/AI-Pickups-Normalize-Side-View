from pickup_measure.src.settings import load_settings


def test_configured_csv_input_path_is_loaded_relative_to_config(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("input_tsv: data/vehicles.csv\n", encoding="utf-8")

    settings = load_settings(config)

    assert settings.input_path == (tmp_path / "data" / "vehicles.csv").resolve()


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


def test_loads_qwen_detection_settings(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(
        "detection:\n"
        "  provider: qwen\n"
        "  model: qwen3-vl-plus\n"
        "  endpoint: https://example.test/v1/chat/completions\n"
        "  api_key: test-secret\n"
        "  prompt_file: custom-prompt.md\n"
        "  timeout_seconds: 45\n"
        "  manual_fallback: false\n",
        encoding="utf-8",
    )

    settings = load_settings(config)

    assert settings.detection.provider == "qwen"
    assert settings.detection.model == "qwen3-vl-plus"
    assert settings.detection.api_key == "test-secret"
    assert settings.detection.prompt_file == (
        tmp_path / "custom-prompt.md"
    ).resolve()
    assert settings.detection.timeout_seconds == 45
    assert settings.detection.manual_fallback is False
    assert settings.detection.perspective_correction is False
