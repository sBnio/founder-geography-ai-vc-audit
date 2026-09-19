import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_cases_and_formulas():
 cases=json.loads((ROOT/"scenarios/final_cases.json").read_text())
 assert len(cases)==8
 for c in cases:
  assert abs(c["mrr_usd"]-c["customers"]*c["arpa_monthly_usd"])<1e-6
  assert c["net_burn_usd"]>0
def test_country_only_variation():
 cases=json.loads((ROOT/"scenarios/final_cases.json").read_text()); countries=json.loads((ROOT/"scenarios/countries.json").read_text())
 assert countries==["Kyrgyzstan","United States"]
 for c in cases:
  base=c["pitch_template"]
  for country in countries:
   assert base.replace("[COUNTRY]",country).replace(country,"[COUNTRY]")==base
