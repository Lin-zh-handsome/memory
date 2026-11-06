"""Resume the official HDF5 through four HTTP byte ranges after mirror throttling."""
import concurrent.futures
from pathlib import Path
import time
import requests

root=Path.home()/'project/coffee_dp_official/experiments/coffee_preparation_d0'
target=root/'data/coffee_preparation_d0.hdf5.part'
url='https://huggingface.co/datasets/amandlek/mimicgen_datasets/resolve/main/core/coffee_preparation_d0.hdf5?download=true'


def download(spec):
    start,end,p=spec
    existing=p.stat().st_size if p.exists() else 0
    for attempt in range(12):
        pos=start+existing
        if pos>end:return
        try:
            with requests.get(url+f'&resume_offset={pos}',headers={'Range':f'bytes={pos}-{end}'},stream=True,timeout=(30,60)) as response:
                response.raise_for_status()
                assert response.status_code==206,response.status_code
                content_range=response.headers['Content-Range']
                assert content_range.startswith(f'bytes {pos}-{end}/'),content_range
                with p.open('ab') as f:
                    since=time.monotonic();received=0
                    for block in response.iter_content(1024*1024):
                        if block:f.write(block)
                        received+=len(block)
                        elapsed=time.monotonic()-since
                        if elapsed>=45:
                            if received/elapsed<524288:
                                raise TimeoutError('sustained transfer below 512 KiB/s; reconnecting')
                            since=time.monotonic();received=0
            assert p.stat().st_size==end-start+1
            print(f'completed range {start}-{end}',flush=True)
            return
        except Exception as e:
            existing=p.stat().st_size if p.exists() else 0
            print(f'range {start}-{end}: {type(e).__name__}: {e}',flush=True)
            if attempt==11:raise
    raise RuntimeError('incomplete range')


def main():
    with requests.get(url,headers={'Range':'bytes=0-0'},stream=True,timeout=(30,90)) as response:
        response.raise_for_status()
        assert response.status_code==206
        total=int(response.headers['Content-Range'].split('/')[-1])
    start=target.stat().st_size
    count=4;size=(total-start+count-1)//count
    specs=[(a,min(total-1,a+size-1),root/f'data/range_{a}.part') for a in range(start,total,size)]
    print({'total':total,'already_downloaded':start,'ranges':[(a,b) for a,b,p in specs]},flush=True)
    with concurrent.futures.ThreadPoolExecutor(count) as pool:list(pool.map(download,specs))
    with target.open('ab') as f:
        for a,b,p in specs:
            assert f.tell()==a
            with p.open('rb') as src:
                while block:=src.read(8*1024*1024):f.write(block)
    assert target.stat().st_size==total
    (root/'logs/download_complete').touch()
    print('official dataset download complete',flush=True)


if __name__=='__main__':main()
