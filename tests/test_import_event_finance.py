import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from import_event_finance import HEADER, parse_csv


class EventFinanceImportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "quantitative.csv"
        self.row = dict.fromkeys(HEADER, "")
        self.row.update({
            "活動日期": "2023-07-05", "活動類型": "大稻埕夏日節",
            "活動狀態": "held", "已舉辦": "1", "已取消": "0", "已延期": "0",
            "實際煙火分鐘": "3", "煙火秒數": "180", "無人機演出": "0",
            "無人機狀態已知": "1", "是否原訂場次": "0",
            "接收合併演出": "0", "接收延期演出": "0",
            "年份": "2023", "月份": "7", "星期序號": "3", "是否週末": "0",
            "相關宣導案所需經費合計(元)": "1592694",
            "平面媒體": "1", "電視媒體": "0", "網路媒體": "1",
            "廣播媒體": "0", "其他": "0",
        })

    def tearDown(self):
        self.temp.cleanup()

    def parse(self, *rows):
        with self.path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HEADER)
            writer.writeheader()
            writer.writerows(rows)
        return parse_csv(self.path)

    def test_annual_amount_is_deduplicated_across_event_dates(self):
        second = dict(self.row, 活動日期="2023-07-12", 星期序號="3")
        events, annual, _ = self.parse(self.row, second)
        self.assertEqual(len(events), 2)
        self.assertEqual(len(annual), 1)
        self.assertEqual(annual[2023]["required_publicity_cost_ntd"], 1_592_694)

    def test_conflicting_annual_amount_rejected(self):
        second = dict(self.row, 活動日期="2023-07-12", 星期序號="3",
                      **{"相關宣導案所需經費合計(元)": "1592695"})
        with self.assertRaisesRegex(ValueError, "Annual fields vary"):
            self.parse(self.row, second)

    def test_missing_publicity_is_unknown_not_zero(self):
        old = dict(self.row, 活動日期="2021-10-16", 活動類型="大稻埕情人節",
                   年份="2021", 月份="10", 星期序號="6", 是否週末="1",
                   **{"相關宣導案所需經費合計(元)": "", "平面媒體": "",
                      "電視媒體": "", "網路媒體": "", "廣播媒體": "", "其他": ""})
        _, annual, _ = self.parse(old)
        self.assertIsNone(annual[2021]["required_publicity_cost_ntd"])


if __name__ == "__main__":
    unittest.main()
