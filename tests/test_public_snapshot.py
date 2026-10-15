"""Public build contract: no database, private inputs, or analysis dependencies."""
import csv
import hashlib
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'app/frontend/public/data'

class PublicSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.obs=json.loads((DATA/'observations.json').read_text(encoding='utf-8'))
        cls.research=json.loads((DATA/'research.json').read_text(encoding='utf-8'))

    def test_public_bundle_hashes(self):
        manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
        self.assertEqual({p.name for p in DATA.iterdir()},set(manifest['outputs'])|{'manifest.json'})
        for name,digest in manifest['outputs'].items():
            self.assertEqual(hashlib.sha256((DATA/name).read_bytes()).hexdigest(),digest,name)

    def test_allowlisted_observation_schema_and_unique_rows(self):
        keys=set()
        for e in self.obs['events']:
            self.assertEqual(set(e),{'event_date','event_name','event_status','announced_show_start_time','fireworks_duration_seconds'})
        for r in self.obs['rows']:
            self.assertEqual(set(r),{'event_date','hour_start','station','origin_count','destination_count_same_source_bin','origin_complete','destination_complete','baseline_origin_mean_28d','baseline_n_days'})
            self.assertIn(r['station'],['北門','大橋頭站','雙連','民權西路'])
            key=(r['event_date'],r['station'],r['hour_start'])
            self.assertNotIn(key,keys);keys.add(key)

    def test_model_metric_matches_exported_forecasts(self):
        m=self.research['model'];h=m['hourly']
        self.assertEqual(len(h),54)
        self.assertAlmostEqual(sum(abs(r['actual']-r['predicted']) for r in h)/len(h),float(m['metrics']['mae']),places=6)
        self.assertEqual({r['hour'] for r in h},set(range(18,24)))

    def test_official_tables_and_results_match_public_sources(self):
        for key in self.research['operations']:
            with (ROOT/f'result/queue_dispatch_v1/metro_{key}.csv').open(encoding='utf-8-sig') as f:
                self.assertEqual(self.research['operations'][key],list(csv.DictReader(f)))
        with (ROOT/'result/queue_dispatch_v1/scenarios.csv').open(encoding='utf-8-sig') as f:
            self.assertEqual(self.research['queue']['scenarios'],list(csv.DictReader(f)))

    def test_csv_download_matches_four_station_observations(self):
        for e in self.obs['events']:
            with (DATA/f"flows-{e['event_date']}.csv").open(encoding='utf-8-sig') as f:
                rows=list(csv.DictReader(f))
            expected=[r for r in self.obs['rows'] if r['event_date']==e['event_date']]
            self.assertEqual(len(rows),len(expected))
            for actual,source in zip(rows,expected):
                self.assertEqual(actual['origin_count'],'' if source['origin_count'] is None else str(source['origin_count']))

if __name__=='__main__': unittest.main()
