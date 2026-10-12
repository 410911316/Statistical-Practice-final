import csv
from datetime import date, datetime, timedelta
from pathlib import Path
import sys
import tempfile
import unittest

import duckdb

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from aggregate_flows import build_analysis
from data_preprocessing import clean_month, validated_clean_cache, HEADER
from export_mysql import sql_value


class CleaningTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.con=duckdb.connect()

    def tearDown(self):
        self.con.close()
        self.tmp.cleanup()

    def write(self,rows):
        p=self.root/"202605.csv"
        with p.open("w",encoding="utf-8-sig",newline="") as f:
            w=csv.writer(f); w.writerow(HEADER); w.writerows(rows)
        return p

    def clean(self,rows):
        return clean_month(self.con,self.write(rows),self.root/"out",self.root/"qa")

    def test_exact_duplicate_deduplicates_but_preserves_zero(self):
        r=self.clean([["2026-05-01","00","A","B","0"]]*2)
        self.assertEqual(r["exact_duplicate_rows_removed"],1)
        self.assertEqual(r["zero_rows"],1)
        self.assertEqual(r["passengers"],0)

    def test_conflicting_duplicate_fails_without_publication(self):
        with self.assertRaisesRegex(ValueError,"conflicting"):
            self.clean([["2026-05-01","00","A","B","0"],
                        ["2026-05-01","00","A","B","7"]])
        self.assertFalse((self.root/"out"/"202605.parquet").exists())

    def test_invalid_count_date_and_hour_fail(self):
        for day,hour,count in [("2026-05-01","00","-1"),("2026-05-01","00","1.5"),
                               ("2026-05-32","00","1"),("2026-06-01","00","1"),
                               ("2026-05-01","24","1")]:
            with self.subTest(day=day,hour=hour,count=count):
                with self.assertRaisesRegex(ValueError,"invalid"):
                    self.clean([[day,hour,"A","B",count]])

    def test_cache_checks_both_source_and_output(self):
        rows=[["2026-05-01","00","A","B","0"]]
        self.clean(rows)
        self.assertTrue(self.clean(rows)["cached"])
        (self.root/"out"/"202605.parquet").write_bytes(b"damaged")
        self.assertNotIn("cached",self.clean(rows))
        self.assertEqual(self.clean([["2026-05-01","00","A","B","4"]])["passengers"],4)

    def test_archived_source_cache_requires_intact_manifest_and_parquet(self):
        self.clean([["2026-05-01","00","A","B","4"]])
        p=self.root/"out"/"202605.parquet"
        self.assertEqual(validated_clean_cache(p,self.root/"qa")["passengers"],4)
        p.write_bytes(b"damaged")
        with self.assertRaisesRegex(ValueError,"validation failed"):
            validated_clean_cache(p,self.root/"qa")

    def test_cross_midnight_baseline_no_future_or_event_leakage(self):
        rows=[]
        # Four-hour windows: previous normal week, prior event, current cancelled event, future.
        for day,amount in [(date(2026,5,1),5),(date(2026,5,8),999),
                           (date(2026,5,15),10),(date(2026,5,22),9999)]:
            for offset in range(22,26):
                stamp=datetime.combine(day,datetime.min.time())+timedelta(hours=offset)
                for origin in ("A","B"):
                    for dest in ("A","B"):
                        rows.append([stamp.date().isoformat(),str(stamp.hour),origin,dest,
                                     str(amount if (origin,dest)==("A","B") else 0)])
        self.clean(rows)
        records=[dict(event_date="2026-05-08",event_name="test",event_status="held"),
                 dict(event_date="2026-05-15",event_name="test",event_status="cancelled")]
        report=build_analysis(self.con,[self.root/"out"/"202605.parquet"],records,
                              self.root/"processed",22,26)
        result=self.con.execute("""SELECT origin_count_window,baseline_origin_window,
          baseline_min_days,observed_event_status,trends_missing FROM regression_event_station
          WHERE event_date='2026-05-15' AND station='A'""").fetchone()
        self.assertEqual(result,(40,20.0,1,"cancelled",True))
        # The following day's 01:00 belongs to the original event date.
        self.assertEqual(self.con.execute("""SELECT origin_count FROM regression_station_hourly
          WHERE event_date='2026-05-15' AND hour_start='2026-05-16 01:00:00' AND station='A'""").fetchone()[0],10)
        self.assertIsNone(self.con.execute("""SELECT origin_count FROM regression_station_hourly
          WHERE event_date='2026-05-15' AND hour_offset=2 AND station='A'""").fetchone()[0])
        self.assertEqual(self.con.execute("""SELECT origin_count_window FROM regression_event_station
          WHERE event_date='2026-05-15' AND station='B'""").fetchone()[0],0)
        self.assertEqual(report["od_passengers"],report["origin_passengers"])
        self.assertEqual(report["od_passengers"],report["destination_passengers"])
        self.assertEqual(self.con.execute("SELECT sum(weight) FROM flow_edges_hourly").fetchone()[0],
                         self.con.execute("SELECT sum(out_strength) FROM flow_nodes_hourly").fetchone()[0])

    def test_incomplete_od_is_null_instead_of_a_partial_total(self):
        self.clean([["2026-05-15","18","A","A","0"],
                    ["2026-05-15","18","B","A","3"],
                    ["2026-05-15","18","B","B","0"]])
        report=build_analysis(self.con,[self.root/"out"/"202605.parquet"],
          [dict(event_date="2026-05-15",event_name="test",event_status="held")],self.root/"processed",18,19)
        self.assertEqual(report["incomplete_od_bins"],1)
        self.assertIsNone(self.con.execute("SELECT origin_count_window FROM regression_event_station WHERE station='A'").fetchone()[0])
        self.assertEqual(self.con.execute("SELECT origin_count_window FROM regression_event_station WHERE station='B'").fetchone()[0],3)

    def test_sql_values_cannot_inject_syntax(self):
        value="北門'); DROP TABLE original_data; --"
        encoded=sql_value(value)
        self.assertNotIn("DROP",encoded)
        self.assertNotIn("'",encoded)
        self.assertEqual(sql_value(None),"NULL")

    def test_cross_month_window_retains_new_station_labels(self):
        clean_paths=[]
        for month,day,hour,stations in [("202604","2026-04-30","23",("A","B")),
                                       ("202605","2026-05-01","00",("B","C"))]:
            p=self.root/f"{month}.csv"
            with p.open("w",encoding="utf-8",newline="") as f:
                w=csv.writer(f); w.writerow(HEADER)
                w.writerows([day,hour,o,d,"1"] for o in stations for d in stations)
            clean_month(self.con,p,self.root/"out",self.root/"qa")
            clean_paths.append(self.root/"out"/f"{month}.parquet")
        report=build_analysis(self.con,clean_paths,
          [dict(event_date="2026-04-30",event_name="test",event_status="held")],self.root/"processed",23,25)
        self.assertEqual(report["event_graph_passengers"],8)
        self.assertEqual(self.con.execute("""SELECT origin_count FROM regression_station_hourly
          WHERE event_date='2026-04-30' AND hour_offset=24 AND station='C'""").fetchone()[0],2)


if __name__=="__main__":
    unittest.main()
