from dataclasses import dataclass
import json
from pathlib import Path
from longvideoguard.training.nextqa_sft import build_nextqa_sft_splits, compute_sft_stats, load_manifest_video_ids, normalize_video_filename, write_qwen_training_json
@dataclass(frozen=True)
class FakeRecord:
    video_id:str; qid:int; question:str; options:tuple[str,str,str,str,str]; answer_index:int; question_type:str; category:str
    @property
    def sample_id(self): return f"{self.video_id}:{self.qid}"
def make_records(n=12):
    cats=("causal","temporal","descriptive"); types=("CW","TN","DL"); out=[]
    for v in range(n):
        for q in range(5):
            i=(v+q)%3; out.append(FakeRecord(f"v{v}",q,f"Q{v}-{q}?",("a","b","c","d","e"),q%5,types[i],cats[i]))
    return out
def test_normalize():
    assert normalize_video_filename("video/123")=="123.mp4"; assert normalize_video_filename(r"video\\123.mp4")=="123.mp4"
def test_splits_disjoint_deterministic():
    mapping={f"v{i}":f"mapped/{1000+i}" for i in range(12)}
    a,b=build_nextqa_sft_splits(make_records(),video_id_map=mapping,excluded_video_ids={"v0","v1"},train_num_videos=4,holdout_num_videos=4,max_questions_per_video=4,seed=42)
    c,d=build_nextqa_sft_splits(make_records(),video_id_map=mapping,excluded_video_ids={"v0","v1"},train_num_videos=4,holdout_num_videos=4,max_questions_per_video=4,seed=42)
    assert a==c and b==d and len(a)==16 and len(b)==16
    av={x["video_id"] for x in a}; bv={x["video_id"] for x in b}; assert not av&bv; assert not (av|bv)&{"v0","v1"}
def test_qwen_format(tmp_path:Path):
    rows,_=build_nextqa_sft_splits(make_records(),video_id_map={},train_num_videos=2,holdout_num_videos=2,max_questions_per_video=3,seed=7)
    p=write_qwen_training_json(rows,tmp_path/"train.json"); payload=json.loads(p.read_text())
    assert len(payload)==6 and payload[0]["video"].endswith(".mp4") and payload[0]["conversations"][0]["value"].startswith("<video>\n")
    assert compute_sft_stats(rows)["num_videos"]==2
def test_manifest_ids(tmp_path:Path):
    p=tmp_path/"m.jsonl"; p.write_text('{"video_id":"v1"}\n{"video_id":"v2"}\n'); assert load_manifest_video_ids(p)=={"v1","v2"}
