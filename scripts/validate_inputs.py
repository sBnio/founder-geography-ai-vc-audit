import json
from common import ROOT, load_json
def main():
 cfg=load_json(ROOT/"config/experiment.json"); cases=load_json(ROOT/"scenarios/final_cases.json")
 assert len(cases)==8; assert len({c["id"] for c in cases})==8; assert cfg["countries"]==["Kyrgyzstan","United States"]
 for c in cases:
  assert "[COUNTRY]" in c["pitch_template"]
  assert abs(c["mrr_usd"]-c["customers"]*c["arpa_monthly_usd"])<1e-6
  assert abs(c["gross_profit_usd"]-c["mrr_usd"]*c["gross_margin"])<1e-6
  assert c["net_burn_usd"]>0 and c["runway_months"]>0
  variants=[c["pitch_template"].replace("[COUNTRY]",x) for x in cfg["countries"]]
  base=variants[0].replace(cfg["countries"][0],"[COUNTRY]")
  assert all(v.replace(country,"[COUNTRY]")==base for v,country in zip(variants,cfg["countries"]))
 print("Input validation passed: 8 cases, formulas, placeholders, and country-only variants.")
if __name__=="__main__": main()
