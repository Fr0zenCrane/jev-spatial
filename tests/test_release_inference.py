from pathlib import Path

import pytest

from spatial_jev.inference import validate_input


@pytest.fixture
def image_file(tmp_path):
    from PIL import Image
    path = tmp_path / 'fixture.png'
    Image.new('RGB', (8, 8), 'white').save(path)
    return str(path)


def request(image_file, space):
    return {'media': [{'kind': 'image', 'uri': image_file}], 'question': 'A spatial question?',
            'answer_space': space}


def test_release_rejects_non_metric_numeric_and_multi_point_contracts(image_file):
    cfg = {'max_images': 4, 'max_choices': 32}
    with pytest.raises(ValueError, match='meters'):
        validate_input(request(image_file, {'kind': 'scalar', 'quantity': 'area', 'unit': 'm2'}), cfg)
    with pytest.raises(ValueError, match='one image'):
        validate_input(request(image_file, {'kind': 'point', 'coordinate_system': 'normalized_xy', 'num_points': 2}), cfg)
    validate_input(request(image_file, {'kind': 'scalar', 'quantity': 'height', 'unit': 'm'}), cfg)


def test_release_rejects_ambiguous_ids_and_missing_media(image_file):
    cfg = {'max_images': 4, 'max_choices': 32}
    value = request(image_file, {'kind': 'choice', 'options': [{'id': 'x', 'text': 'left'}, {'id': 'x', 'text': 'right'}]})
    with pytest.raises(ValueError, match='unique'):
        validate_input(value, cfg)
    value['media'][0]['uri'] = image_file + '.absent'
    with pytest.raises(FileNotFoundError):
        validate_input(value, cfg)


def test_clean_runtime_has_no_training_or_adapter_imports():
    import ast
    import spatial_jev.runtime as runtime
    tree = ast.parse(Path(runtime.__file__).read_text())
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert not any(name and ('peft' in name or 'unified' in name or 'train' in name) for name in imports)
    assert not hasattr(runtime.InferenceBuilder, 'training_example')
