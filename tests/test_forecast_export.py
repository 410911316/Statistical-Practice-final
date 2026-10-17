"""Exercise the migrated export against the existing public snapshot in a temp folder."""
from pathlib import Path
import json,tempfile,unittest,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
class SnapshotRepository:
    def __init__(self):
        self.data=json.loads((ROOT/'app/frontend/public/data/observations.json').read_text('utf8'))
    def health(self):return {}
    def events(self):return self.data['events']
    def flows(self,date,stations,start,end):
        return [r for r in self.data['rows'] if r['event_date']==str(date) and r['station'] in stations]
class ExportContract(unittest.TestCase):
    @unittest.skipUnless((ROOT/'.local/forecast/predictions.csv').exists(),'Private forecast required')
    def test_migrated_export_preserves_public_predictions_and_queue(self):
        from app.export_snapshot import export
        old=json.loads((ROOT/'app/frontend/public/data/research.json').read_text('utf8'))
        with tempfile.TemporaryDirectory() as tmp:
            export(SnapshotRepository(),Path(tmp))
            new=json.loads((Path(tmp)/'research.json').read_text('utf8'))
            self.assertEqual(new['queue'],old['queue'])
            self.assertEqual(new['operations'],old['operations'])
            self.assertEqual(new['model']['events'],old['model']['events'])
            self.assertEqual(len(new['model']['hourly']),54)
            for a,b in zip(new['model']['hourly'],old['model']['hourly']):
                self.assertEqual(a['event_date'],b['event_date'])
                self.assertEqual(a['hour'],b['hour'])
                self.assertEqual(a['actual'],b['actual'])
                self.assertAlmostEqual(a['predicted'],b['predicted'],places=8)
            replay=json.loads((Path(tmp)/'rail-replay.json').read_text('utf8'))
            previous=json.loads((ROOT/'app/frontend/public/data/rail-replay.json').read_text('utf8'))
            self.assertEqual(replay['events'],previous['events'])
if __name__=='__main__':unittest.main()
