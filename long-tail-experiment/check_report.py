"""Static report integrity; optionally compare bytes served by the preview."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import argparse,json,subprocess,urllib.request
parser=argparse.ArgumentParser();parser.add_argument('--url');args=parser.parse_args()
P=Path(__file__).resolve().parent;report=P.parent/'mlp-length-experiment/report.html'
class Check(HTMLParser):
    def __init__(self):
        super().__init__();self.ids=[];self.links=[];self.tables=[];self.row=None;self.table=None
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a':self.links.append(a.get('href',''))
        if tag=='table':self.table=[]
        if tag=='tr':self.row=0
        if tag in ('th','td'):self.row+=1
    def handle_endtag(self,tag):
        if tag=='tr':self.table.append(self.row);self.row=None
        if tag=='table':self.tables.append(self.table);self.table=None
c=Check();c.feed(report.read_text());assert len(c.ids)==len(set(c.ids)) and 'long-tail' in c.ids
for table in c.tables:assert len(set(table))==1,table
for href in c.links:
    u=urlsplit(href)
    if u.scheme or u.netloc:continue
    q=(report.parent/unquote(u.path)).resolve() if u.path else report.resolve()
    assert q.is_file(),href
    if u.fragment and q==report.resolve():assert u.fragment in c.ids,href
for path in P.iterdir():
    if path.suffix not in ('.py','.c','.sh','.txt','.md','.json'):continue
    r=subprocess.run(['git','diff','--no-index','--check','/dev/null',str(path)],capture_output=True,text=True)
    assert not r.stdout and not r.stderr,(path,r.stdout,r.stderr)
if args.url:assert urllib.request.urlopen(args.url).read()==report.read_bytes()
r={'passed':True,'tables':len(c.tables),'data_rows_per_table':[len(t)-1 for t in c.tables],'links_checked':len(c.links),'ids_checked':len(c.ids),'served_bytes_equal':True if args.url else None,'new_source_whitespace_check':True}
(P/'logs/html-static.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
