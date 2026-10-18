"""Validate the whole retained report and sixth-stage evidence links."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import argparse,json,subprocess,urllib.request
P=Path(__file__).resolve().parent;report=P.parent/'mlp-length-experiment/report.html';parser=argparse.ArgumentParser();parser.add_argument('--url');a=parser.parse_args()
class Check(HTMLParser):
 def __init__(self):super().__init__();self.ids=[];self.hrefs=[];self.tables=[];self.table=None;self.row=None
 def handle_starttag(self,tag,attrs):
  v=dict(attrs)
  if 'id' in v:self.ids.append(v['id'])
  if tag=='a' and 'href' in v:self.hrefs.append(v['href'])
  if tag=='table':self.table=[]
  if tag=='tr':self.row=0
  if tag in ('th','td'):self.row+=1
 def handle_endtag(self,tag):
  if tag=='tr':self.table.append(self.row);self.row=None
  if tag=='table':self.tables.append(self.table);self.table=None
text=report.read_text();c=Check();c.feed(text);assert len(c.ids)==len(set(c.ids)) and all(x in c.ids for x in ('bf16-pipeline','fusion-pipeline','long-tail'))
for table in c.tables:assert len(set(table))==1,table
for href in c.hrefs:
 u=urlsplit(href)
 if u.scheme or u.netloc:continue
 target=(report.parent/unquote(u.path)).resolve() if u.path else report.resolve();assert target.is_file(),href
 if u.fragment and target==report.resolve():assert u.fragment in c.ids,href
for f in P.iterdir():
 if f.suffix not in ('.py','.c','.cpp','.inc','.sh','.txt','.md','.json'):continue
 r=subprocess.run(['git','diff','--no-index','--check','/dev/null',str(f)],capture_output=True,text=True);assert not r.stdout and not r.stderr,(f,r.stdout,r.stderr)
s=json.loads((P/'logs/summary.json').read_text());section=text.split('<section id="bf16-pipeline"',1)[1].split('</section>',1)[0];assert f"{s['candidate']['median']*1000:.2f}" in section and f"{s['candidate']['p95_nearest_rank']*1000:.2f}" in section
assert ('目标达标' if s['goal_met'] else '未达到1秒目标') in section
links=[x for x in c.hrefs if '../bf16-pipeline-experiment/' in x];assert len(links)>=25
if a.url:assert urllib.request.urlopen(a.url).read()==report.read_bytes()
r={'passed':True,'tables':len(c.tables),'ids':len(c.ids),'all_links_checked':len(c.hrefs),'sixth_stage_links':len(links),'served_bytes_equal':True if a.url else None,'source_whitespace_passed':True};(P/'logs/report-check.json').write_text(json.dumps(r,indent=2)+'\n');print('REPORT CHECK',r)
