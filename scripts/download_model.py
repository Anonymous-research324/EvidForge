"""Fetch the official ModelScope files and enforce the paper's model hashes."""
from pathlib import Path
import argparse,hashlib,json,urllib.request,urllib.parse,os
R=Path(__file__).resolve().parents[1]
def digest(path):
 h=hashlib.sha256()
 with path.open("rb") as stream:
  for b in iter(lambda:stream.read(4*1024*1024),b""):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",default=str(R/"models/Qwen3.5-9B"));p.add_argument("--revision");a=p.parse_args()
 origin=json.loads((R/"provenance/model_source.json").read_text());revision=a.revision or origin["revision"];folder=Path(a.output);folder.mkdir(parents=True,exist_ok=True)
 for line in (R/"provenance/model-files-9B.sha256").read_text().splitlines():
  expected,name=line.split()
  if Path(name).name!=name:raise ValueError("Model manifest must contain base filenames")
  target=folder/name
  if target.exists():
   if digest(target)!=expected:raise ValueError("Existing model file has different hash; inspect "+str(target))
   print("VERIFIED",name,flush=True);continue
  query=urllib.parse.urlencode({"Revision":revision,"FilePath":name})
  url="https://modelscope.cn/api/v1/models/"+origin["model_id"]+"/repo?"+query
  temporary=target.with_name(target.name+".part");h=hashlib.sha256()
  request=urllib.request.Request(url,headers={"User-Agent":"EvidForge-paper-reproduction/1.0"})
  print("DOWNLOAD",name,flush=True)
  with urllib.request.urlopen(request,timeout=120) as response,temporary.open("wb") as stream:
   while True:
    block=response.read(4*1024*1024)
    if not block:break
    stream.write(block);h.update(block)
  if h.hexdigest()!=expected:raise ValueError("Official snapshot differs from paper hashes; retain .part and use matching historical revision")
  os.replace(temporary,target)
 print("All model files match the paper manifest.")
if __name__=="__main__":main()
