"""Product-quality gates: never substitute the small source-format fixtures."""

import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

pytestmark = pytest.mark.real_data


def test_real_structure_and_source_alignment(real_store, real_source):
    train, yt, st = real_store.arrays('train')
    test, y, s = real_store.arrays('test')
    assert train.shape == (7352, 561)
    assert test.shape == (2947, 561)
    assert set(st).isdisjoint(s)
    assert len(set(st) | set(s)) == 30
    assert set(yt) == set(y) == set(range(1, 7))
    assert [f['feature_id'] for f in real_store.features()] == [f'f{i:03}' for i in range(1, 562)]
    with real_source.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == real_store.info()['dataset_id']
    with zipfile.ZipFile(real_source) as outer:
        nested = next(n for n in outer.namelist() if n.lower().endswith('.zip'))
        payload = outer.read(nested)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        prefix = next(n[:-len('features.txt')] for n in archive.namelist() if n.endswith('/features.txt'))
        for split, count in [('train', 7352), ('test', 2947)]:
            def original(name, split=split):
                return np.loadtxt(io.BytesIO(archive.read(prefix + split + '/' + name)))
            x = original(f'X_{split}.txt')
            labels = original(f'y_{split}.txt')
            subjects = original(f'subject_{split}.txt')
            axes = {axis: original(f'Inertial Signals/total_acc_{axis}_{split}.txt') for axis in 'xyz'}
            for index in (0, count - 1):
                sample = real_store.sample(f'{split}:{index + 1:06}')
                np.testing.assert_array_equal(list(sample['features'].values()), x[index])
                assert sample['actual_label'] == labels[index]
                assert sample['subject_id'] == subjects[index]
                np.testing.assert_array_equal(sample['signal']['time_seconds'], np.arange(128) / 50)
                for axis in 'xyz':
                    np.testing.assert_array_equal(sample['signal'][axis], axes[axis][index].astype(np.float32).astype(float))


def test_all_ready_real_model_reports_and_reload_predictions(real_store, real_registry):
    x, y, _ = real_store.arrays('test')
    ids = [f['feature_id'] for f in real_store.features()]
    models = [m for m in real_registry.list() if m['status'] == 'ready'
              and m['dataset_id'] == real_store.info()['dataset_id']]
    assert {'full', 'reduced'} <= {m['profile'] for m in models if m['trees'] == 50 and m['seed'] == 42}
    for model in models:
        artifact = real_registry.load(model['model_id'])
        columns = [ids.index(fid) for fid in model['feature_ids']]
        frame = pd.DataFrame(x[:, columns], columns=model['feature_ids'])
        predicted = artifact['estimator'].predict(frame)
        reloaded = real_registry.load(model['model_id'])['estimator'].predict(frame)
        np.testing.assert_array_equal(predicted, reloaded)
        report = real_registry.report(model['model_id'])
        assert report['test_count'] == 2947
        assert report['accuracy'] == pytest.approx(accuracy_score(y, predicted))
        assert report['macro_f1'] == pytest.approx(f1_score(y, predicted, average='macro'))
        np.testing.assert_array_equal(report['confusion_matrix'], confusion_matrix(y, predicted, labels=range(1, 7)))
        p, r, f, support = precision_recall_fscore_support(y, predicted, labels=range(1, 7), zero_division=0)
        for i, cls in enumerate(report['per_class']):
            assert cls['label_id'] == i + 1
            assert cls['precision'] == pytest.approx(p[i])
            assert cls['recall'] == pytest.approx(r[i])
            assert cls['f1'] == pytest.approx(f[i])
            assert cls['support'] == support[i]
