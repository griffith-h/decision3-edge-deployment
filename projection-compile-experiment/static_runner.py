"""Specialize warm signatures; unseen signatures use the original CPU path."""
import os,runpy,sys
os.environ['D3_COMPILE_DYNAMIC']='0'
script=sys.argv.pop(1)
runpy.run_path('/work/'+script,run_name='__main__')
