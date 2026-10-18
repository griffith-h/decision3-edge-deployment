"""Check the complete report and new evidence without rewriting prior audits."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import argparse,json,subprocess,urllib.request
P=Path(__file__).resolve().parent;report=P.parent/'mlp-length-experiment/report.html'
parser=argparse.ArgumentParser();parser.add_argument('--url');args=parser.parse_args()
class Check(HTMLParser):
 def __init__(self):super().__init__();self.ids=[];self.hrefs=[];self.tables=[];self.row=None;self.table=None
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if 'id' in a:self.ids.append(a['id'])
  if tag=='a' and 'href' in a:self.hrefs.append(a['href'])
  if tag=='table':self.table=[]
  if tag=='tr':self.row=0
  if tag in ('th','td'):self.row+=1
 def handle_endtag(self,tag):
  if tag=='tr':self.table.append(self.row);self.row=None
  if tag=='table':self.tables.append(self.table);self.table=None
c=Check();c.feed(report.read_text());assert len(c.ids)==len(set(c.ids));assert 'fusion-pipeline' in c.ids and 'long-tail' in c.ids
for table in c.tables:assert len(set(table))==1,table
links=[x for x in c.hrefs if '../fusion-pipeline-experiment/' in x];assert len(links)>=20
for x in c.hrefs:
 u=urlsplit(x)
 if u.scheme or u.netloc:continue
 target=(report.parent/unquote(u.path)).resolve() if u.path else report.resolve();assert target.is_file(),x
 if u.fragment and target==report.resolve():assert u.fragment in c.ids,x
for path in P.iterdir():
 if path.suffix not in ('.py','.c','.cpp','.inc','.sh','.txt','.md','.json'):continue
 r=subprocess.run(['git','diff','--no-index','--check','/dev/null',str(path)],capture_output=True,text=True);assert not r.stdout and not r.stderr,(path,r.stdout,r.stderr)
s=json.loads((P/'logs/summary.json').read_text());text=report.read_text();assert f"{s['candidate']['median']*1000:.2f}" in text;assert ('目标达标' if s['goal_met'] else '未达到1秒目标') in text
if args.url:assert urllib.request.urlopen(args.url).read()==report.read_bytes()
audit={'passed':True,'tables':len(c.tables),'data_rows_per_table':[len(t)-1 for t in c.tables],'all_links_checked':len(c.hrefs),'fifth_stage_links':len(links),'ids_checked':len(c.ids),'served_bytes_equal':True if args.url else None,'root_source_whitespace_check':True}
(P/'logs/html-static-all-stages.json').write_text(json.dumps(audit,indent=2)+'\n');print('REPORT PASS',json.dumps(audit))
