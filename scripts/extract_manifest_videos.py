from __future__ import annotations
import argparse, json, shutil, zipfile
from pathlib import Path, PurePosixPath
VIDEO_SUFFIXES={".mp4",".avi",".mov",".mkv",".webm"}
def load_names(manifests):
    names=set()
    for manifest in manifests:
        for line_number,line in enumerate(manifest.read_text(encoding="utf-8").splitlines(),1):
            if line.strip():
                row=json.loads(line); name=PurePosixPath(str(row.get("video_relpath","")).replace("\\","/")).name
                if not name: raise ValueError(f"{manifest}:{line_number} has no video_relpath")
                names.add(name)
    if not names: raise ValueError("No videos found")
    return names
def main() -> int:
    p=argparse.ArgumentParser(description="Extract videos referenced by JSONL manifests")
    p.add_argument("archive",type=Path); p.add_argument("manifests",type=Path,nargs="+")
    p.add_argument("--output-dir",type=Path,default=Path("data/raw/nextqa/sft_videos")); p.add_argument("--overwrite",action="store_true")
    a=p.parse_args(); names=load_names(a.manifests); a.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(a.archive) as z:
        index={}
        for info in z.infolist():
            member=PurePosixPath(info.filename)
            if not info.is_dir() and member.suffix.lower() in VIDEO_SUFFIXES: index.setdefault(member.stem,[]).append(info)
        extracted=skipped=0; missing=[]; ambiguous=[]
        for pos,name in enumerate(sorted(names),1):
            matches=index.get(PurePosixPath(name).stem,[])
            if not matches: missing.append(name); continue
            if len(matches)>1:
                exact=[x for x in matches if PurePosixPath(x.filename).name==name]
                if len(exact)!=1: ambiguous.append(name); continue
                info=exact[0]
            else: info=matches[0]
            dst=a.output_dir/name
            if dst.exists() and not a.overwrite: skipped+=1; print(f"[{pos}/{len(names)}] SKIP {dst}"); continue
            tmp=dst.with_suffix(dst.suffix+".part"); tmp.unlink(missing_ok=True); print(f"[{pos}/{len(names)}] GET {info.filename}")
            with z.open(info) as source, tmp.open("wb") as target: shutil.copyfileobj(source,target,1024*1024)
            tmp.replace(dst); extracted+=1; print(f"[{pos}/{len(names)}] OK {dst}")
    summary={"expected":len(names),"extracted":extracted,"skipped":skipped,"missing":missing,"ambiguous":ambiguous}; print(json.dumps(summary,indent=2)); return 1 if missing or ambiguous else 0
if __name__=="__main__": raise SystemExit(main())
