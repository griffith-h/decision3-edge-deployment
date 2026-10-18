"""Check fifth-stage evidence links and report structure without inference."""
from pathlib import Path
from html.parser import HTMLParser
import json
P=Path(__file__).resolve().parent;report=P.parent/'mlp-length-experiment/report.html'
class Check(HTMLParser):
 def __init__(self):super().__init__();self.ids=[];self.hrefs=[];self.tables=0
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id' in a:self.ids.append(a['id'])
  if tag=='a' and 'href' in a:self.hrefs.append(a['href'])
  if tag=='table':self.tables+=1
c=Check();c.feed(report.read_text());assert len(c.ids)==len(set(c.ids));assert 'fusion-pipeline' in c.ids and 'long-tail' in c.ids
links=[x for x in c.hrefs if '../fusion-pipeline-experiment/' in x]
assert len(links)>=20
for x in links:assert (report.parent/x.split('#')[0]).exists(),x
s=json.loads((P/'logs/summary.json').read_text());text=report.read_text();assert f"{s['candidate']['median']*1000:.2f}" in text;assert ('目标达标' if s['goal_met'] else '未达到1秒目标') in text
print('REPORT PASS',len(links),'fifth-stage links',c.tables,'total tables')
