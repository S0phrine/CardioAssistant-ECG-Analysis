"""Integration checks using supplied weights and synthetic WFDB recordings."""
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import wfdb

from app.app_gui import CardioAssistantApp, load_metadata, load_state_dict, predict_probabilities
from cardio_ml.cnn import ECG1DCNN, build_rr_feature
from cardio_ml.database import initialize_database, save_analysis
from cardio_ml.data import load_feature_dataset
from cardio_ml.segments import load_segment_dataset

ROOT = Path(__file__).resolve().parents[1]

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.meta = load_metadata(ROOT / 'models/best_model.meta.json')
        cls.model = ECG1DCNN()
        cls.model.load_state_dict(load_state_dict(ROOT / 'models/best_model.pt', torch.device('cpu')), strict=True)
        cls.model.eval()

    def test_weights_predictions_are_repeatable_and_batch_independent(self):
        x = np.random.default_rng(42).normal(size=(7, 180)).astype(np.float32)
        rr = np.linspace(-1, 1, 7).astype(np.float32)
        a, _ = predict_probabilities(self.model, x, rr, torch.device('cpu'), 2)
        b, _ = predict_probabilities(self.model, x, rr, torch.device('cpu'), 7)
        np.testing.assert_allclose(a, b, atol=1e-6)
        np.testing.assert_allclose(a.sum(axis=1), 1, atol=1e-6)

    def test_wfdb_to_predictions_database_and_excel(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            samples = np.arange(3600)
            peaks = np.array([360, 650, 1000, 1370, 1700, 2100, 2500, 2890, 3250])
            signal = sum(np.exp(-((samples - p) / 9) ** 2) for p in peaks)
            wfdb.wrsamp('100', fs=360, units=['mV'], sig_name=['ECG'], p_signal=signal[:, None], fmt=['16'], write_dir=d)
            wfdb.wrann('100', 'atr', sample=peaks, symbol=['N', 'A', 'V'] * 3, write_dir=d)
            ds = load_segment_dataset(path, ['100'])
            features = load_feature_dataset(path, ['100'])
            self.assertEqual(ds.signals.shape, (9, 180))
            self.assertEqual(features.features.shape, (9, 12))
            np.testing.assert_allclose(features.features[:, -1], ds.rr_previous, atol=1e-6)
            with self.assertRaises(ValueError):
                load_segment_dataset(path, ['100'], expected_sampling_frequency=250)
            rr = (build_rr_feature(ds.rr_previous, ds.rr_next, 'ratio') - self.meta['rr_norm_mu'][0]) / self.meta['rr_norm_sd'][0]
            probs, preds = predict_probabilities(self.model, ds.signals, rr, torch.device('cpu'))
            segment = dict(start=0, end=8, beat_count=9, probabilities=probs, predictions=preds, counts={'N':9}, mean_confidence=float(probs.max(axis=1).mean()), records='100')
            db = path / 'result.db'
            initialize_database(db)
            save_analysis(segment, ('N','S','V'), 'best_model.pt', db)
            with sqlite3.connect(db) as con:
                self.assertEqual(con.execute('select count(*) from heartbeat').fetchone()[0], 9)
                self.assertEqual(con.execute('select typeof(probability_n) from heartbeat limit 1').fetchone()[0], 'real')
            invalid = {**segment, 'predictions':np.full(9, 8)}
            with self.assertRaises(ValueError):
                save_analysis(invalid, ('N','S','V'), 'bad', db)
            with sqlite3.connect(db) as con:
                self.assertEqual(con.execute('select count(*) from analysis').fetchone()[0], 1)
            app = SimpleNamespace(last_analysis=segment, class_names=('N','S','V'))
            CardioAssistantApp.export_to_excel(app, path / 'result.xlsx')
            self.assertEqual(len(pd.read_excel(path / 'result.xlsx', sheet_name='Uderzenia')), 9)

    def test_training_export_loads_in_application(self):
        # Training deliberately selects the noninteractive plotting backend.
        from experiments.train_cnn import save_results
        config = dict(input_channels=2, sampling_frequency_hz=360, half_window_samples=90, rr_mean=1.0, rr_standard_deviation=0.5, rr_feature='ratio', first_layer_channels=96, dropout=0.2, seed=864)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            save_results(self.model, p, config, np.array([0,1,2]), np.array([0,1,2]), np.eye(3))
            meta = load_metadata(p / 'checkpoint.meta.json')
            self.assertEqual(meta['in_ch'], 2)
            restored = ECG1DCNN()
            restored.load_state_dict(load_state_dict(p / 'checkpoint.pt', torch.device('cpu')), strict=True)
            for key, val in self.model.state_dict().items():
                torch.testing.assert_close(val, restored.state_dict()[key])

if __name__ == '__main__':
    unittest.main()
